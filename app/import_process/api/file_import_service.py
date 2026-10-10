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
    3. 多文件批量上传时每个文件一个 task_id，互不影响。

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
from fastapi.responses import FileResponse

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
    update_task_status,
)

# 允许上传的文件后缀（与导入节点能力匹配：PDF 走转换，MD 直接处理）
ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown"}

# 单文件大小上限，与 MinerU 侧限制保持一致
MAX_FILE_SIZE_MB = 100

# 定义 fastapi 对象
app = FastAPI(title="file import service", description="掌柜智库文件导入服务！")

# 跨域配置：allow_credentials=True 时不能用通配符 *（浏览器会拒绝），
# 这里用 regex 匹配任意来源，等效放开且不触发 CORS 报错
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
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
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 -> 本地保存 -> 启动后台任务
# 访问地址：http://localhost:8000/upload （POST 请求，form-data 格式传参）
# --------------------------
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    文件上传核心接口（不上传 MinIO）

    1. 接收前端上传的多文件（PDF/MD 为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一 task_id，启动独立的 LangGraph 后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI 后台任务对象，用于异步执行 LangGraph 流程
    :param files: 前端上传的文件列表（form-data 格式）
    :return: 包含上传结果和所有任务 ID 的 JSON 响应
    """
    if not files:
        raise HTTPException(status_code=400, detail="未接收到任何文件")

    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个 task_id）
    task_ids: List[str] = []
    details: List[Dict[str, Any]] = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成 task_id）
    for file in files:
        # 生成全局唯一 task_id（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 文件名做安全处理：只保留纯文件名，防止 ../ 路径穿越
        safe_name = os.path.basename(file.filename or f"unnamed_{task_id}")
        suffix = Path(safe_name).suffix.lower()

        # 校验文件后缀与大小
        if suffix not in ALLOWED_SUFFIXES:
            add_done_task(task_id, "upload_file", status="failed")
            logger.warning(f"[{task_id}] 文件类型不支持：{suffix}，已跳过")
            details.append({"filename": safe_name, "accepted": False,
                            "reason": f"不支持的文件类型 {suffix}，仅支持 {sorted(ALLOWED_SUFFIXES)}"})
            task_ids.pop()  # 未受理的任务不返回 task_id，避免前端轮询一个永远不动的任务
            continue

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/task_id，避免多文件重名冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / safe_name

        # 6. 将上传的文件保存到本地临时目录（分块复制，避免大文件占满内存）
        try:
            max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            written = 0
            with local_file_abs_path.open("wb") as file_buffer:
                shutil.copyfileobj(file.file, file_buffer)
            written = local_file_abs_path.stat().st_size
            if written > max_bytes:
                raise ValueError(f"文件大小 {written / 1024 / 1024:.1f}MB 超过上限 {MAX_FILE_SIZE_MB}MB")
        except Exception as exc:
            add_done_task(task_id, "upload_file", status="failed")
            update_task_status(task_id, "failed")
            logger.error(f"[{task_id}] 文件保存失败：{exc}")
            task_ids.pop()
            details.append({"filename": safe_name, "accepted": False, "reason": f"保存失败：{exc}"})
            continue

        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}（{written} 字节）")

        # 7. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file")
        details.append({"filename": safe_name, "accepted": True, "size": written})

        # 8. 将 LangGraph 全流程处理加入 FastAPI 后台任务
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path),
        )
        logger.info(f"[{task_id}] 已将 LangGraph 全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有 task_id
    logger.info(f"多文件上传处理完毕，共受理 {len(task_ids)} 个文件，生成 task_id 列表：{task_ids}")
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
        host="127.0.0.1",  # 仅本地访问，生产环境改为 0.0.0.0（允许所有 IP 访问）
        port=8010,  # 服务端口（8000 已被 Milvus Attu 占用，默认改用 8010）
    )