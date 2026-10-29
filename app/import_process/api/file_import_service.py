"""
文件导入 Web 服务（FastAPI）

把 LangGraph 知识库导入流程包装成 HTTP 服务：
    - POST /upload          接收文件上传，后台触发全流程导入，立刻返回 task_id
    - GET  /status/{task_id} 轮询任务进度（已完成/运行中节点列表 + 全局状态）
    - GET  /documents       已入库文档列表（MongoDB 台账，可按需从 Milvus 回填）
    - DELETE /documents/{doc_id} 删除文档（Milvus 切片 + MinIO 图片 + 台账，真删）
    - GET  /import.html     返回前端上传页面

设计要点：
    1. 导入是耗时的（PDF转MD + 视觉模型 + 向量化 + 入库），故用 BackgroundTasks
       放到后台执行，避免阻塞 HTTP 响应；
    2. 进度通过 task_utils 的内存态对外暴露，前端轮询 /status 即可看到逐节点进度；
    3. 多文件批量上传时每个主文档一个 task_id，互不影响；图片作为附属资源
       复制进各 MD 任务的 images/ 目录，不单独成任务。

启动方式：
    python -m app.import_process.api.file_import_service
    # 然后访问 http://127.0.0.1:8010/import.html
"""
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

# 【必须放在最前】限制 MKL/OpenMP 线程数，早于任何 torch 初始化，
# 否则 BGE-M3 向量化会 Segmentation fault。原理见 app/core/torch_env.py。
# 这里用 as 起个别名，只为避免与下方模块级的 FastAPI 实例 app 同名引起歧义
import app.core.torch_env as _torch_env  # noqa: F401  导入即生效

from app.core.logger import PROJECT_ROOT, logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.nodes.node_import_milvus import _derive_doc_id
from app.import_process.agent.state import get_default_state
from app.clients.minio_config import minio_config
from app.clients.minio_utils import delete_objects_by_prefix
from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import get_milvus_client
from app.clients.mongo_document_utils import (
    count_documents,
    delete_document,
    get_document,
    list_documents,
    upsert_document,
)
from app.utils.access_guard import verify_write_request
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.tenant_guard import require_tenant_filter, resolve_tenant
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_failed_task_list,
    get_global_status,
    get_running_task_list,
    set_default_namespace,
    update_task_status,
)

# P1-2：导入任务的 task_id 语义独立于查询会话，设置专属命名空间，
# 避免与查询服务的 session_id 在同一进程内互相覆盖。
set_default_namespace("import")

# 主文档后缀：一个主文档对应一个独立导入任务
MAIN_DOC_SUFFIXES = {".pdf", ".md", ".markdown"}

# 附属图片后缀：随 MD 文档一起上传，落进 md 同级 images/ 供 node_md_img 扫描。
# 刻意不含 .svg —— 可携带脚本，而对象桶曾长期匿名可读。
ATTACHMENT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# node_md_img 约定的图片目录名，须与 md_file.parent / "images" 保持一致
IMAGES_SUBDIR = "images"

# 图片真实格式魔数，防止仅靠后缀改名绕过
_IMAGE_MAGIC_PREFIXES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")

# 单文件大小上限，与 MinerU 侧限制保持一致
MAX_FILE_SIZE_MB = 100

# 定义 fastapi 对象
app = FastAPI(title="file import service", description="掌柜智库文件导入服务！")

# 跨域配置（P2-4）：allow_credentials=True 时不能用通配符 *（浏览器会拒绝），
# 且通配 ".*" 等于允许任意站点携带凭证访问，风险过大。
# 改为白名单：来源从环境变量 ALLOWED_ORIGINS 读取（逗号分隔），
# 默认仅放行本机前端开发地址，部署时按实际域名配置。
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8011,http://localhost:8011",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# 静态页面路由：返回文件导入前端页面 import.html
# 访问地址：http://127.0.0.1:8010/import.html
# --------------------------
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """
    返回文件导入前端页面：import.html

    :return: HTML 文件响应；文件不存在时抛 404
    """
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出 404 异常
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # 以 FileResponse 返回 HTML 文件，浏览器自动渲染
    # 注意变量名不要叫 app —— 它会遮蔽模块级的 FastAPI 实例 app，
    # 在此函数内再想拿app（如挂路由/加中间件）就会拿到 FileResponse 对象
    return FileResponse(
        path=str(html_abs_path),
        media_type="text/html",  # 显式指定媒体类型为 HTML，确保浏览器正确解析
    )


# --------------------------
# 根路径：重定向到前端页面，避免直接访问 / 时刷 404
# --------------------------
@app.get("/", include_in_schema=False)
async def root_redirect():
    """根路径重定向到文件导入前端页面 import.html"""
    logger.info("根路径访问，重定向到 /import.html")
    return RedirectResponse(url="/import.html")


# --------------------------
# 站点图标：浏览器加载页面会自动请求 /favicon.ico，
# 未提供图标时返回 204（无内容），避免日志刷 404
# --------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """返回空 204 响应，消除 favicon 请求产生的 404 噪音"""
    return Response(status_code=204)


# --------------------------
# 健康检查
# --------------------------
@app.get("/health", summary="健康检查", description="确认服务与 Milvus/图是否可用")
async def health_check():
    """轻量健康检查，不触发完整的导入流程"""
    from app.clients.milvus_utils import get_milvus_client
    from app.clients.milvus_config import milvus_config

    milvus_ok = get_milvus_client() is not None
    return {
        "code": 200,
        "status": "ok",
        "milvus_connected": milvus_ok,
        "chunks_collection": milvus_config.chunks_collection,
        "graph_nodes": len(kb_import_app.get_graph().nodes),
    }


# --------------------------
# 后台任务：LangGraph 全流程执行
# 独立于主请求线程，由 BackgroundTasks 触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str, tenant_id: str):
    """
    LangGraph 全流程执行后台任务

    核心流程：初始化状态 -> 流式执行图节点 -> 实时更新任务状态 -> 异常捕获
    任务状态更新：pending -> processing -> completed/failed
    节点进度更新：每完成一个节点，将节点名记录到 task_utils，供前端轮询查看

    注意：各业务节点内部已各自调用 add_running_task / add_done_task，
         这里额外再记一次是为了在「节点抛异常」时也能把状态落到 failed
         （业务节点只在自己成功路径上报 done，异常路径未必执行到）。

    :param task_id: 全局唯一任务 ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    :param tenant_id: 租户标识，来自已验签的内部令牌，全链路唯一的租户来源
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行 LangGraph 全流程，本地文件路径：{local_file_path}")

        # 2. 初始化 LangGraph 状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["tenant_id"] = tenant_id  # 租户归属，决定写进哪个 partition
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # 3. 流式执行 LangGraph 全流程
        #用 stream_mode="values" 拿到每次迭代后的完整状态，这样能区分出
        #「哪些键是本节点新增/更新的」，从而准确判断刚执行完的是哪个节点。
        #（默认 stream_mode="updates" 只返回增量，用 list(event.keys()) 取节点名
        #   会把 langgraph 内部键也带出来，这里统一用 values 更稳妥）
        last_keys = set(init_state.keys())
        final_state: Dict[str, Any] = {}
        for event in kb_import_app.stream(init_state, stream_mode="values"):
            if not event:
                continue
            final_state = event
            # 新出现的键就是本节点写入的字段，据此反推节点名
            new_keys = [k for k in event.keys() if k not in last_keys]
            node_hint = "graph"
            for candidate in ("pdf_path", "md_content", "item_name", "chunks"):
                if candidate in new_keys:
                    node_hint = {
                        "pdf_path": "node_pdf_to_md",
                        "md_content": "node_document_split",
                        "item_name": "node_item_name_recognition",
                        "chunks": "node_bge_embedding",
                    }[candidate]
                    break
            logger.info(f"[{task_id}] LangGraph 节点执行完成：{node_hint}")
            add_done_task(task_id, node_hint)
            last_keys = set(event.keys())

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        # 台账落库：文档已经在 Milvus 里了，这里补一条「它存在」的索引，
        # 供导入页的文档列表与删除使用。写失败只记日志，不影响导入结果。
        _record_document(task_id, tenant_id, final_state)
        logger.info(f"[{task_id}] LangGraph 全流程执行完毕，任务完成")

    except Exception as exc:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph 全流程执行失败，异常信息：{str(exc)}", exc_info=True)


# --------------------------
# 文档台账：记录「库里有哪些文档」，供导入页的文档列表与删除使用
# --------------------------
def _record_document(task_id: str, tenant_id: str, final_state: Dict[str, Any]) -> None:
    """
    导入成功后补写一条文档台账

    doc_id 必须与 Milvus 切片同源：切片侧由 node_import_milvus 按
    sha1(tenant_id|file_title) 派生，这里直接复用切片上已回填的 doc_id，
    取不到时再用同一个函数现算，保证「台账里的 doc_id」永远能删到「切片里的 doc_id」。
    两边算法一旦不一致，删除就会静默删 0 条 —— 所以宁可依赖上游也不自己拼。

    写台账失败不算导入失败：文档已经在 Milvus 里了，缺一条索引还能靠 /documents?sync=true 补回来。

    :param task_id: 导入任务 ID
    :param tenant_id: 租户标识
    :param final_state: LangGraph 跑完后的完整状态
    :return: None
    """
    try:
        chunks = final_state.get("chunks") or []
        if not chunks:
            logger.warning(f"[{task_id}] 台账未写入：final_state 中没有切片")
            return

        file_title = (
            final_state.get("file_title")
            or chunks[0].get("file_title")
            or ""
        ).strip()
        doc_id = (chunks[0].get("doc_id") or "").strip() or _derive_doc_id(tenant_id, file_title)

        upsert_document(
            doc_id=doc_id,
            tenant_id=tenant_id,
            file_title=file_title,
            item_name=final_state.get("item_name") or "",
            chunk_count=len(chunks),
            task_id=task_id,
            imported_at=datetime.now().timestamp(),
            source="import",
        )
    except Exception as exc:
        # 台账只是索引，不能让它把「已经成功的导入」顶成失败
        logger.error(f"[{task_id}] 台账写入异常（不影响导入结果）：{exc}", exc_info=True)


def _scan_milvus_documents(tenant_id: str) -> List[Dict[str, Any]]:
    """
    从 Milvus 反查某租户下有哪些文档（按 doc_id 聚合）

    只在台账为空（升级前入库的老数据）或显式要求同步时调用：全量扫切片代价不低，
    正常路径应该走台账。分批用 query_iterator 拉，避免一次性把大集合读进内存。

    :param tenant_id: 租户标识
    :return: [{doc_id, file_title, item_name, chunk_count}, ...]；失败返回空列表
    """
    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.warning("Milvus 不可用，跳过文档同步")
        return []

    expr = require_tenant_filter(tenant_id)
    output_fields = ["doc_id", "file_title", "item_name"]
    grouped: Dict[str, Dict[str, Any]] = {}

    def _collect(rows):
        for row in rows or []:
            doc_id = (row.get("doc_id") or "").strip()
            if not doc_id:
                # 早期数据可能没有 doc_id 字段（动态字段缺失），用 file_title 现算，
                # 保证它仍然是一个「可被删除」的键
                doc_id = _derive_doc_id(tenant_id, row.get("file_title") or "")
            entry = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "file_title": row.get("file_title") or "",
                    "item_name": row.get("item_name") or "",
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1

    try:
        iterator = milvus_client.query_iterator(
            collection_name=milvus_config.chunks_collection,
            batch_size=1000,
            filter=expr,
            output_fields=output_fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                _collect(batch)
        finally:
            # 不 close 会占着 Milvus 的服务端资源，长时间跑会累积
            iterator.close()
    except Exception as exc:
        # 老版本 pymilvus 没有 query_iterator，退回单次 query（受 16384 上限约束）
        logger.warning(f"query_iterator 不可用，退回单次查询：{exc}")
        try:
            rows = milvus_client.query(
                collection_name=milvus_config.chunks_collection,
                filter=expr,
                output_fields=output_fields,
                limit=16384,
            )
            _collect(rows)
        except Exception as inner:
            logger.error(f"从 Milvus 反查文档失败：{inner}")
            return []

    logger.info(f"从 Milvus 反查到 {len(grouped)} 份文档（tenant={tenant_id}）")
    return list(grouped.values())


def _sync_documents_from_milvus(tenant_id: str) -> int:
    """
    用 Milvus 反查结果回填台账（幂等，重复执行只会刷新 updated_at）

    回填记录的 imported_at 传 None：Milvus 切片里没有导入时间，
    编一个时间出来会让「按导入时间排序」变成假排序。

    :param tenant_id: 租户标识
    :return: 回填条数
    """
    documents = _scan_milvus_documents(tenant_id)
    synced = 0
    for doc in documents:
        if upsert_document(
            doc_id=doc["doc_id"],
            tenant_id=tenant_id,
            file_title=doc["file_title"],
            item_name=doc["item_name"],
            chunk_count=doc["chunk_count"],
            task_id="",
            imported_at=None,
            source="sync",
        ):
            synced += 1
    logger.info(f"台账回填完成：{synced}/{len(documents)} 条")
    return synced


def _inspect_milvus_document(tenant_id: str, doc_id: str) -> Dict[str, Any]:
    """
    从 Milvus 查某文档的切片条数与来源文件名

    删除前先探一遍：既是为了向用户说清「删掉了几条」，也是为了在台账缺失时
    仍能拿到 file_title 去清 MinIO 图片。探不到就返回空结果，不阻塞真正的删除。

    :param tenant_id: 租户标识
    :param doc_id: 文件级幂等键
    :return: {"chunk_count": int, "file_title": str}；查不到时 chunk_count 为 0
    """
    empty = {"chunk_count": 0, "file_title": ""}
    milvus_client = get_milvus_client()
    if milvus_client is None:
        return empty
    expr = _document_filter(tenant_id, doc_id)
    try:
        rows = milvus_client.query(
            collection_name=milvus_config.chunks_collection,
            filter=expr,
            output_fields=["count(*)"],
        )
        chunk_count = int(rows[0].get("count(*)", 0) or 0) if rows else 0
    except Exception as exc:
        logger.warning(f"统计切片条数失败（doc_id={doc_id[:8]}…）：{exc}")
        chunk_count = 0

    file_title = ""
    try:
        # 取任意一条即可：同一 doc_id 下所有切片的 file_title 都相同
        rows = milvus_client.query(
            collection_name=milvus_config.chunks_collection,
            filter=expr,
            output_fields=["file_title"],
            limit=1,
        )
        if rows:
            file_title = (rows[0].get("file_title") or "").strip()
    except Exception as exc:
        logger.warning(f"查询文档来源文件名失败（doc_id={doc_id[:8]}…）：{exc}")
    return {"chunk_count": chunk_count, "file_title": file_title}


def _document_filter(tenant_id: str, doc_id: str) -> str:
    """
    拼「本租户 + 本文件」的 Milvus 过滤表达式

    租户条件必须由 require_tenant_filter 生成（它会在租户为空时直接抛错），
    再与 doc_id 用 and 连接，避免手拼字符串时漏掉租户把别人的数据删了。

    :param tenant_id: 租户标识
    :param doc_id: 文件级幂等键
    :return: 可直接用于 query / delete 的表达式
    """
    return (
        f'{require_tenant_filter(tenant_id)} '
        f'and doc_id == "{escape_milvus_string(doc_id)}"'
    )


# --------------------------
# 上传落盘工具
# --------------------------
def _write_upload(source: UploadFile, dest_path: Path) -> int:
    """
    将上传流写入目标路径，并校验大小上限

    每次写入前 seek(0)：同一张图片需要复制进多个任务目录，流会被重复读取。

    :param source: 上传文件对象
    :param dest_path: 目标绝对路径
    :return: 写入字节数
    :raises ValueError: 超过 MAX_FILE_SIZE_MB（此时已删除落盘文件）
    """
    source.file.seek(0)
    with dest_path.open("wb") as file_buffer:
        shutil.copyfileobj(source.file, file_buffer)

    written = dest_path.stat().st_size
    if written > MAX_FILE_SIZE_MB * 1024 * 1024:
        dest_path.unlink(missing_ok=True)
        raise ValueError(f"文件大小 {written / 1024 / 1024:.1f}MB 超过上限 {MAX_FILE_SIZE_MB}MB")
    return written


def _is_real_image(source: UploadFile) -> bool:
    """
    按魔数判断是否真为图片，避免改名绕过后缀白名单

    :param source: 上传文件对象（调用后流位置回到开头）
    :return: 是受支持图片格式返回 True
    """
    source.file.seek(0)
    head = source.file.read(16)
    source.file.seek(0)
    if head.startswith(_IMAGE_MAGIC_PREFIXES):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 一次请求内：主文档各自成任务，图片作为附属资源复制进每个 md 任务
# 访问地址：http://localhost:8000/upload （POST 请求，form-data 格式传参）
# --------------------------
@app.post(
    "/upload",
    summary="文件上传接口",
    description="上传主文档（可附带图片），自动触发知识库导入全流程",
    dependencies=[Depends(verify_write_request)],
)
async def upload_files(
    background_tasks: BackgroundTasks,
    request: Request,
    files: List[UploadFile] = File(...),
):
    """
    文件上传核心接口（不上传 MinIO）

    1. 按后缀分流：PDF/MD 为主文档，图片为附属资源，其余拒绝
    2. 每个主文档生成唯一 task_id，保存到 output/YYYYMMDD/{task_id}/
    3. 附属图片复制到 md 任务的 {task_id}/images/，与 node_md_img 的扫描约定对齐；
       PDF 的 images/ 由 MinerU 转换产出，不接受附属图片
    4. 为每个主文档启动独立 LangGraph 后台任务，状态供前端轮询

    :param background_tasks: FastAPI 后台任务对象，用于异步执行 LangGraph 流程
    :param request: 当前请求，用于解析租户
    :param files: 前端上传的文件列表（form-data 格式）
    :return: 包含上传结果和所有任务 ID 的 JSON 响应
    """
    # 租户只来自已验签的内部令牌；独立本机模式下落到固定的 local 租户
    tenant_id = resolve_tenant(request)

    if not files:
        raise HTTPException(status_code=400, detail="未接收到任何文件")

    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 2. 分流：主文档 / 附属图片 / 不支持
    docs: List[UploadFile] = []
    images: List[UploadFile] = []
    details: List[Dict[str, Any]] = []
    for file in files:
        safe_name = os.path.basename(file.filename or "unnamed")
        suffix = Path(safe_name).suffix.lower()
        if suffix in MAIN_DOC_SUFFIXES:
            docs.append(file)
        elif suffix in ATTACHMENT_IMAGE_SUFFIXES:
            images.append(file)
        else:
            details.append({"filename": safe_name, "accepted": False,
                            "reason": f"不支持的文件类型 {suffix}，"
                                      f"文档仅支持 {sorted(MAIN_DOC_SUFFIXES)}，"
                                      f"图片仅支持 {sorted(ATTACHMENT_IMAGE_SUFFIXES)}"})

    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"未接收到主文档，仅支持 {sorted(MAIN_DOC_SUFFIXES)}；图片只能作为附属文件随文档上传",
        )

    # 3. 附属图片先落盘到暂存目录并校验魔数，再按任务复制：
    #    一张坏图只报告一次，而不是每个文档任务各报一遍
    task_ids: List[str] = []
    staged_images: List[Path] = []
    staging_dir = date_based_root_dir / f"_attach_{uuid.uuid4()}"
    if images:
        staging_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            img_name = os.path.basename(image.filename or f"image_{index}")
            if not _is_real_image(image):
                details.append({"filename": img_name, "accepted": False,
                                "reason": "文件内容与图片魔数不符，已拒绝"})
                logger.warning(f"附属图片魔数校验失败，已拒绝：{img_name}")
                continue
            try:
                _write_upload(image, staging_dir / img_name)
                staged_images.append(staging_dir / img_name)
            except Exception as exc:
                details.append({"filename": img_name, "accepted": False, "reason": f"图片保存失败：{exc}"})
                logger.error(f"附属图片保存失败：{img_name}，原因：{exc}")

    # 4. 逐个主文档建任务
    for doc in docs:
        task_id = str(uuid.uuid4())
        safe_name = os.path.basename(doc.filename or f"unnamed_{task_id}")
        suffix = Path(safe_name).suffix.lower()
        logger.info(f"[{task_id}] 开始处理上传文档，文件名：{safe_name}，类型：{doc.content_type}")

        add_running_task(task_id, "upload_file")
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        try:
            written = _write_upload(doc, task_local_dir / safe_name)
        except Exception as exc:
            add_done_task(task_id, "upload_file", status="failed")
            update_task_status(task_id, "failed")
            logger.error(f"[{task_id}] 文档保存失败：{exc}")
            details.append({"filename": safe_name, "accepted": False, "reason": f"保存失败：{exc}"})
            continue

        # 5. 附属图片进 md 同级 images/，node_md_img 才会扫描到
        attached = 0
        if staged_images:
            if suffix in {".md", ".markdown"}:
                images_dir = task_local_dir / IMAGES_SUBDIR
                images_dir.mkdir(parents=True, exist_ok=True)
                for staged in staged_images:
                    try:
                        shutil.copy2(staged, images_dir / staged.name)
                        attached += 1
                    except Exception as exc:
                        logger.error(f"[{task_id}] 图片复制失败 {staged.name}：{exc}")
                logger.info(f"[{task_id}] 附属图片已写入 {images_dir}（{attached} 张）")
            else:
                logger.warning(f"[{task_id}] PDF 的 images/ 由 MinerU 转换产出，忽略 {len(staged_images)} 张附属图片")

        task_ids.append(task_id)
        add_done_task(task_id, "upload_file")
        details.append({"filename": safe_name, "accepted": True, "size": written, "images_attached": attached})

        # 6. 将 LangGraph 全流程处理加入 FastAPI 后台任务
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(task_local_dir / safe_name),
            tenant_id,
        )
        logger.info(f"[{task_id}] 已将 LangGraph 全流程加入后台任务，任务已启动")

    # 7. 图片已复制进各任务目录，暂存目录即刻回收
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info(f"上传处理完毕，共受理 {len(task_ids)} 个文档任务，task_id 列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(task_ids)}",
        "task_ids": task_ids,
        "details": details,
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度
# 访问地址：http://localhost:8000/status/{task_id} （GET 请求）
# --------------------------
@app.get(
    "/status/{task_id}",
    summary="任务状态查询",
    description="根据 task_id 查询单个文件的处理进度和全局状态",
    # 原先这个接口完全没有校验，任何人拿到/猜到 task_id 就能读别人的导入进度与
    # 文件名。合并部署模式下必须带合法内部令牌才放行（独立模式沿用本机校验，不破坏老用法）。
    dependencies=[Depends(resolve_tenant)],
)
async def get_task_progress(task_id: str):
    """
    任务状态查询接口

    前端轮询此接口（如每 2 秒一次），获取任务的实时处理进度。
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无 IO。

    :param task_id: 全局唯一任务ID（由 /upload 接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的 JSON 响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        # 任务全局状态：pending / processing / completed / failed
        "status": get_global_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "failed_list": get_failed_task_list(task_id),
    }
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，"
        f"已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info


# --------------------------
# 核心接口：已入库文档管理
# 原先只有 /upload（写）和 /status（看进度），没有「库里现在有哪些文档」的视角——
# 任务进度是内存态，服务一重启就空了，用户看不到历史导入结果，也没法纠正导错的资料。
# 这里补一张基于 MongoDB 台账的文档视图，并提供真删入口。
# --------------------------
# 每个租户只自动回填一次，避免台账为空时每次打开页面都全量扫一遍 Milvus
_auto_synced_tenants: set = set()


@app.get(
    "/documents",
    summary="已入库文档列表",
    description="列出当前租户已入库的文档，支持按需从 Milvus 回填台账",
    dependencies=[Depends(resolve_tenant)],
)
async def list_documents_api(request: Request, sync: bool = False):
    """
    已入库文档列表

    数据来自 MongoDB 台账（kb_documents），不是内存任务表，所以服务重启后依然能看。
    台账为空（升级前入库的老数据）时自动从 Milvus 回填一次；
    想强制重新对齐可以带 ?sync=true。

    :param request: 当前请求（用于解析租户）
    :param sync: 是否强制从 Milvus 回填台账
    :return: 文档列表与总条数
    """
    tenant_id = resolve_tenant(request)

    existing = count_documents(tenant_id)
    if sync or (existing == 0 and tenant_id not in _auto_synced_tenants):
        synced = _sync_documents_from_milvus(tenant_id)
        _auto_synced_tenants.add(tenant_id)
        logger.info(f"文档台账回填 {synced} 条（tenant={tenant_id}, reason={'force' if sync else 'empty'}）")

    rows = list_documents(tenant_id)
    return {
        "code": 200,
        "tenant": tenant_id,
        "total": len(rows),
        "documents": rows,
    }


@app.delete(
    "/documents/{doc_id}",
    summary="删除已入库文档",
    description="真删：Milvus 切片 + MinIO 图片 + 台账记录，不可恢复",
    # 删除是不可逆操作，必须过写接口来源校验（本机页面 + 可选 X-API-Key）
    dependencies=[Depends(verify_write_request)],
)
async def delete_document_api(doc_id: str, request: Request):
    """
    删除一份已入库文档（真删）

    三步都要做，少一步就留垃圾：
        1. Milvus 切片：按 tenant_id + doc_id 删，租户条件由 require_tenant_filter 兜底，
           不会漏租户把别人同名文档删掉；
        2. MinIO 图片：按 upload-images/{file_title}/ 前缀清，文档删了图片留着，
           下次同名导入会误用旧图；
        3. 台账记录：最后删，前两步失败时至少还能重试。

    第 2 步失败只记日志不中断：切片删掉后文档已经检索不到了，
    此时因为图片没清干净就报失败，会让用户重复点删除却看不到任何变化。

    :param doc_id: 文件级幂等键（来自 /documents 返回的 doc_id）
    :param request: 当前请求（用于解析租户）
    :return: 各环节实际删除数量
    """
    tenant_id = resolve_tenant(request)
    doc_id = (doc_id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id 不能为空")

    record = get_document(doc_id, tenant_id)
    inspected = _inspect_milvus_document(tenant_id, doc_id)
    if not record and inspected["chunk_count"] == 0 and not inspected["file_title"]:
        # 不区分「不存在」和「属于别的租户」：后者不该被探测出来
        raise HTTPException(status_code=404, detail="文档不存在，或不属于当前租户")

    file_title = (record or {}).get("file_title") or inspected["file_title"]

    # 1. 删 Milvus 切片
    deleted_chunks = 0
    milvus_client = get_milvus_client()
    if milvus_client is not None and inspected["chunk_count"] > 0:
        try:
            milvus_client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=_document_filter(tenant_id, doc_id),
            )
            deleted_chunks = inspected["chunk_count"]
            # 删除在 Milvus 里是打标记，flush 后才对检索稳定生效
            try:
                milvus_client.flush(collection_name=milvus_config.chunks_collection)
            except Exception as exc:
                logger.warning(f"删除后 flush 失败（不改变已删除状态）：{exc}")
        except Exception as exc:
            logger.error(f"删除 Milvus 切片失败（doc_id={doc_id[:8]}…）：{exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"删除向量切片失败：{exc}") from exc

    # 2. 清 MinIO 图片（尽力而为）
    deleted_images = 0
    if file_title:
        prefix = f"{minio_config.minio_img_dir}/{file_title}/"
        deleted_images = delete_objects_by_prefix(prefix)

    # 3. 删台账
    deleted_records = delete_document(doc_id, tenant_id)

    logger.info(
        f"文档已删除：{file_title or doc_id[:8]}（tenant={tenant_id}, "
        f"切片={deleted_chunks}, 图片={deleted_images}, 台账={deleted_records}）"
    )
    return {
        "code": 200,
        "doc_id": doc_id,
        "file_title": file_title,
        "deleted_chunks": deleted_chunks,
        "deleted_images": deleted_images,
        "deleted_records": deleted_records,
    }


# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动 FastAPI 服务，无需额外执行 uvicorn 命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动 uvicorn 服务，绑定本地 IP 和 8000 端口，关闭自动重载（生产环境建议用 workers 多进程）
    # 第一个参数用位置传递（不用 app=app），避免与模块级 app 变量在阅读时产生歧义
    uvicorn.run(
        app,  # FastAPI 实例
        host=os.getenv("APP_HOST", "127.0.0.1"),  # docker 内通过 APP_HOST=0.0.0.0 对外暴露
        port=8010,  # 服务端口（8000 已被 Milvus Attu 占用，默认改用 8010）
    )