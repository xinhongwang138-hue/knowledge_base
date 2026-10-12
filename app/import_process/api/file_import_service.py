"""
文件导入 Web 服务（FastAPI）

把 LangGraph 知识库导入流程包装成 HTTP 服务：
    - POST /upload          接收文件上传，后台触发全流程导入，立刻返回 task_id
    - GET  /status/{task_id} 轮询任务进度（已完成/运行中节点列表 + 全局状态）
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
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

# 【必须放在最前】限制 MKL/OpenMP 线程数，早于任何 torch 初始化，
# 否则 BGE-M3 向量化会 Segmentation fault。原理见 app/core/torch_env.py。
# 这里用 as 起个别名，只为避免与下方模块级的 FastAPI 实例 app 同名引起歧义
import app.core.torch_env as _torch_env  # noqa: F401  导入即生效

from app.core.logger import PROJECT_ROOT, logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
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
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
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
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行 LangGraph 全流程，本地文件路径：{local_file_path}")

        # 2. 初始化 LangGraph 状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # 3. 流式执行 LangGraph 全流程
        #用 stream_mode="values" 拿到每次迭代后的完整状态，这样能区分出
        #「哪些键是本节点新增/更新的」，从而准确判断刚执行完的是哪个节点。
        #（默认 stream_mode="updates" 只返回增量，用 list(event.keys()) 取节点名
        #   会把 langgraph 内部键也带出来，这里统一用 values 更稳妥）
        last_keys = set(init_state.keys())
        for event in kb_import_app.stream(init_state, stream_mode="values"):
            if not event:
                continue
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
        logger.info(f"[{task_id}] LangGraph 全流程执行完毕，任务完成")

    except Exception as exc:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph 全流程执行失败，异常信息：{str(exc)}", exc_info=True)


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
@app.post("/upload", summary="文件上传接口", description="上传主文档（可附带图片），自动触发知识库导入全流程")
async def upload_files(
    background_tasks: BackgroundTasks,
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
    :param files: 前端上传的文件列表（form-data 格式）
    :return: 包含上传结果和所有任务 ID 的 JSON 响应
    """
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
@app.get("/status/{task_id}", summary="任务状态查询", description="根据 task_id 查询单个文件的处理进度和全局状态")
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