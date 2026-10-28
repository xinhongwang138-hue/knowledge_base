"""
检索查询 Web 服务（FastAPI）

把检索主图 query_app 包装成可被浏览器使用的聊天服务：

    - GET    /chat.html        返回前端聊天页面
    - POST   /query            提交问题，后台执行检索图；流式模式立即返回 session_id
    - GET    /stream/{session_id}  SSE 长连接，实时推送进度 / 增量文本 / 最终答案
    - GET    /history/{session_id} 读取该会话的历史对话（来自 MongoDB）
    - DELETE /history/{session_id} 清空该会话历史
    - GET    /health           健康检查

四个模块的协作关系（笔记 8.2.3 的分层）：
    Web 服务层(本文件)  接收请求、建立 SSE 连接
    SSE 工具层(sse_utils) 管理消息队列、打包与推送事件
    任务状态层(task_utils) 记录节点进度，并触发 progress 事件
    图节点层(query_process/) 实际业务逻辑，通过更新 state 驱动进度

流式 vs 非流式：
    - 流式(is_stream=True)：后台跑图，边跑边通过 SSE 推delta，前端有打字机效果
    - 非流式(is_stream=False)：同步跑完再一次性返回 answer，适合程序调用
"""
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# 【必须放在最前】限制 MKL 线程数，早于任何 torch 初始化（检索会用到 BGE-M3）
import app.core.torch_env as _torch_env  # noqa: F401  导入即生效

from app.utils.access_guard import verify_write_request
from app.utils.tenant_guard import resolve_tenant
from app.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
)
from app.clients.minio_utils import presign_public_image_url
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    push_progress,
    push_to_session,
    sse_generator,
)
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_failed_task_list,
    get_running_task_list,
    get_task_result,
    set_default_namespace,
    set_task_result,
    update_task_status,
)

# P1-2：查询会话的 task_id 语义独立于导入任务，设置专属命名空间，
# 避免与导入服务的 task_id 在同一进程内互相覆盖。
set_default_namespace("query")

# 定义 fastapi 对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

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


# ===================== 精排模型启动预热 =====================
# 冷启动首问曾出现「单次精排 13.4s」：一是 Cross-Encoder 权重首次加载，
# 二是首次前向的算子/内存分配抖动。两者都与具体问题无关，属于可提前支付的成本。
# 因此把这段开销从「用户第一问」挪到「服务启动」：后台线程预热一次单条前向。
# 约束：预热失败只记日志、绝不阻断启动；首次精排仍会懒加载重试，可用性不受影响。
# 开关：RERANK_WARMUP=false 可关闭（本地调试、极低内存环境）。
_RERANK_WARMUP_STARTED = False


def _rerank_warmup_enabled() -> bool:
    raw = os.getenv("RERANK_WARMUP")
    if raw is None or raw == "":
        return True
    return raw.strip().strip('"').strip("'").lower() in ("true", "1", "yes", "on")


def _warmup_reranker() -> None:
    """加载 reranker 权重并跑一次单条前向，使模型常驻、算子上下文就绪。"""
    try:
        from app.lm.reranker_utils import rerank_scores

        started = time.time()
        scores = rerank_scores(
            "预热问题：产品保修期多久",
            ["预热文档：本条仅用于触发模型加载与首次前向，不参与真实检索。"],
        )
        logger.info(
            f"Rerank 预热完成，耗时 {time.time() - started:.1f}s，"
            f"得分 {[round(s, 4) for s in scores]}"
        )
    except Exception as exc:  # noqa: BLE001 预热属尽力而为，失败不影响服务
        logger.warning(f"Rerank 预热失败（不影响服务，首次精排会重新懒加载）：{exc}")


@app.on_event("startup")
def _on_startup_warmup() -> None:
    """服务启动后异步预热精排模型，不阻塞端口监听。"""
    global _RERANK_WARMUP_STARTED
    if _RERANK_WARMUP_STARTED:
        return
    if not _rerank_warmup_enabled():
        logger.info("RERANK_WARMUP 已关闭，跳过精排预热")
        return
    _RERANK_WARMUP_STARTED = True
    threading.Thread(target=_warmup_reranker, name="rerank-warmup", daemon=True).start()
    logger.info("已在后台线程启动 Rerank 预热")


# ===================== 请求/响应模型 =====================
class QueryRequest(BaseModel):
    """查询请求数据结构"""

    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


# ===================== 页面路由 =====================
@app.get("/chat.html")
async def chat():
    """
    返回检索聊天前端页面 chat.html

    :return: HTML 文件响应；文件不存在时抛 404
    """
    # api 目录的上一级是 query_process，页面在其 page 子目录下
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        logger.error(f"页面不存在：{chat_html_path}")
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    logger.info("成功加载 chat.html 页面")
    return FileResponse(str(chat_html_path), media_type="text/html")


# ===================== 核心：后台执行检索图 =====================
def _push_progress_event(session_id: str, is_stream: bool) -> None:
    """
    把当前节点进度推给前端（流式模式才有意义）

    :param session_id: 会话唯一标识
    :param is_stream: 是否流式
    """
    if not is_stream:
        return
    push_progress(
        session_id,
        done_list=get_done_task_list(session_id),
        running_list=get_running_task_list(session_id),
        failed_list=get_failed_task_list(session_id),
    )


def run_query_graph(
    session_id: str,
    user_query: str,
    is_stream: bool = True,
    tenant_id: str = "",
) -> Dict[str, Any]:
    """
    后台执行检索图（LangGraph）

    负责：
        1. 记录用户提问到 MongoDB
        2. 执行 query_app，过程中持续把节点进度推给 SSE
        3. 把最终答案存进 task_result（供非流式接口同步读取）
        4. 异常时更新任务状态并推送 error 事件

    :param session_id: 会话唯一标识
    :param user_query: 用户问题
    :param is_stream: 是否流式推送
    :return: 最终状态摘要（answer / item_names / done_list）
    """
    logger.info(f"[{session_id}] 开始执行查询流程，流式模式：{is_stream}")
    update_task_status(session_id, TASK_STATUS_PROCESSING)
    _push_progress_event(session_id, is_stream)

    final_state: Dict[str, Any] = {}
    try:
        # 【历史存档职责划分】——存档全部由节点层负责，本层【不重复保存】：
        #   · 用户提问 → node_item_name_confirm 存档，且它会带着识别出的
        #     item_names 回来回填同一条消息（见该节点 step_7），
        #     服务层再存一份会导致「一问一答」在MongoDB 里落两套重复记录。
        #   · 助手回答 → node_answer_output 存档，因为只有它知道 image_urls。
        init_state = create_query_default_state(
            session_id=session_id,
            original_query=user_query,
            is_stream=is_stream,
            # 租户标识必须由接口层解析后注入：检索节点用它拼强制过滤表达式，
            # 为空会让「向量检索」「HyDE 检索」双双抛错降级为空结果，
            # 表现为本地召回恒为空、误触发联网兜底（P0-3）。
            tenant_id=tenant_id,
        )

        # 逐节点推进：每完成一个节点就推一次进度
        last_keys = set(init_state.keys())
        for event in query_app.stream(init_state, stream_mode="values"):
            if not event:
                continue
            final_state = event
            new_keys = [k for k in event.keys() if k not in last_keys]
            if new_keys:
                # 有新字段写入说明有新节点产出，推一次进度
                _push_progress_event(session_id, is_stream)
            last_keys = set(event.keys())

        answer = final_state.get("answer", "")
        item_names = final_state.get("item_names", [])
        # 引用溯源（P0-1）：答案节点产出的来源清单，随任务结果一并缓存
        sources = final_state.get("sources") or []

        # 结果落到任务缓存，供非流式接口读取
        set_task_result(
            session_id,
            answer=answer,
            item_names=item_names,
            sources=sources,
            done_list=get_done_task_list(session_id),
        )

        update_task_status(session_id, TASK_STATUS_COMPLETED)
        _push_progress_event(session_id, is_stream)
        logger.info(f"[{session_id}] 查询流程执行完成，答案长度：{len(answer)}")

    except Exception as exc:
        logger.error(f"[{session_id}] 查询流程异常：{exc}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_FAILED)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(exc)})
        set_task_result(session_id, answer="", error=str(exc))

    return {
        "answer": final_state.get("answer", ""),
        "item_names": final_state.get("item_names", []),
        "done_list": get_done_task_list(session_id),
    }


# ===================== 查询接口 =====================
@app.post("/query", dependencies=[Depends(verify_write_request)])
async def query(
    background_tasks: BackgroundTasks,
    payload: QueryRequest,
    request: Request,
):
    """
    提交检索查询

    流式(is_stream=True)：创建 SSE 队列 → 后台跑图 → 立即返回 session_id，
                        前端拿 session_id 去连 /stream/{session_id} 接收增量推送
    非流式(is_stream=False)：同步跑完再返回完整答案

    :param background_tasks: FastAPI 后台任务对象
    :param payload: 查询请求体
    :param request: 原始 HTTP 请求，用于解析租户标识（合并部署校验内部令牌）
    :return: 流式返回 {"message","session_id"}；非流式额外带 answer / done_list
    """
    user_query = (payload.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    session_id = payload.session_id or str(uuid.uuid4())
    is_stream = payload.is_stream
    # 租户解析放在入口处一次性完成：独立本机模式回退 local，
    # 合并部署时缺令牌直接 401/403，不把越权风险带到检索层。
    tenant_id = resolve_tenant(request)
    logger.info(
        f"[{session_id}] 收到查询：{user_query}（流式={is_stream}, tenant={tenant_id}）"
    )

    if is_stream:
        # 队列必须先于后台任务创建，否则图推事件时还没有监听者
        create_sse_queue(session_id)
        logger.info(f"[{session_id}] 已创建 SSE 消息队列")
        update_task_status(session_id, TASK_STATUS_PROCESSING)
        background_tasks.add_task(
            run_query_graph, session_id, user_query, is_stream, tenant_id
        )
        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {"message": "结果正在处理中...", "session_id": session_id}

    # 非流式：同步执行
    run_query_graph(session_id, user_query, is_stream, tenant_id)
    return {
        "message": "处理完成！",
        "session_id": session_id,
        "answer": get_task_result(session_id, "answer", ""),
        "item_names": get_task_result(session_id, "item_names", []),
        # 引用溯源（P0-1）：与答案 [i] 序号一一对应的来源清单
        "sources": get_task_result(session_id, "sources", []),
        "done_list": get_done_task_list(session_id),
    }


# ===================== SSE 流式推送接口 =====================
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """
    SSE 长连接，实时接收该会话的检索进度与答案增量

    事件类型：ready / progress / delta / final / error

    :param session_id: 会话唯一标识
    :param request: Request 对象，用于检测客户端断开
    :return: StreamingResponse（text/event-stream）
    """
    logger.info(f"[{session_id}] 客户端已建立 SSE 流式连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            # 关闭 nginx 缓冲，否则事件会被攒起来一次性下发，打字机失效
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===================== 预签名重签辅助 =====================
_MD_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")


def _presign_text_image_urls(text: str) -> str:
    """将文本内 Markdown 图片 URL 实时重签为预签名 URL（P1-5）。"""
    if not text:
        return text

    def _replace(match):
        url = match.group(1).strip()
        new_url = presign_public_image_url(url)
        if new_url == url:
            return match.group(0)
        whole = match.group(0)
        alt = whole[2:whole.index("](")]
        return f"![{alt}]({new_url})"

    return _MD_IMAGE_PATTERN.sub(_replace, text)


# ===================== 历史对话接口 =====================
@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录

    :param session_id: 会话唯一标识
    :param limit: 返回条数上限
    :return: {"session_id": ..., "items": [...]}
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = [
            {
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                # P1-5：历史落库文本内嵌的图片 URL 可能已过期，返回前实时重签
                "text": _presign_text_image_urls(r.get("text", "")),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                # P1-5：历史落库的图片直链在私有桶下不可匿名访问，
                # 返回前实时重签为预签名 URL（字段名与结构保持不变）
                "image_urls": [
                    presign_public_image_url(url) or url
                    for url in (r.get("image_urls") or [])
                ],
                # 引用溯源（P0-1）：历史回看时还原「来源」区块
                "sources": r.get("sources") or [],
                "ts": r.get("ts"),
            }
            for r in records
        ]
        return {"session_id": session_id, "items": items}
    except Exception as exc:
        logger.error(f"[{session_id}] 查询历史失败：{exc}")
        raise HTTPException(status_code=500, detail=f"history error: {exc}")


@app.delete("/history/{session_id}", dependencies=[Depends(verify_write_request)])
async def clear_chat_history(session_id: str):
    """
    清空该会话的全部历史记录

    :param session_id: 会话唯一标识
    :return: {"message": "History cleared", "deleted_count": N}
    """
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


# ===================== 健康检查 =====================
@app.get("/health")
async def health():
    """
    服务健康检查：MongoDB 与 Milvus 任一不可用即判定为不健康

    说明：旧实现无论依赖是否可用都硬编码 ok=True，导致依赖断开时
    容器仍显示 healthy，故障被掩盖。现在
      - ok = mongo_connected and milvus_connected
      - 依赖异常时返回 HTTP 503，与 compose healthcheck 的 ok 字段校验配套

    :return: 各依赖的连通状态
    """
    mongo_ok = False
    mongo_detail = "未配置"
    try:
        from app.clients.mongo_history_utils import get_history_mongo_tool

        mongo_ok = bool(get_history_mongo_tool().ping())
        mongo_detail = "正常" if mongo_ok else "ping 返回 False"
    except Exception as exc:
        mongo_detail = str(exc)[:120]

    milvus_ok = False
    milvus_detail = "未配置"
    try:
        from app.clients.milvus_utils import get_milvus_client

        # list_collections 真实访问 Milvus 元数据服务，比看进程存活可靠
        get_milvus_client().list_collections()
        milvus_ok = True
        milvus_detail = "正常"
    except Exception as exc:
        milvus_detail = str(exc)[:120]

    ok = bool(mongo_ok and milvus_ok)
    logger.info(f"健康检查：mongo={mongo_ok} milvus={milvus_ok} ok={ok}")
    payload = {
        "ok": ok,
        "mongo_connected": mongo_ok,
        "mongo_detail": mongo_detail,
        "milvus_connected": milvus_ok,
        "milvus_detail": milvus_detail,
        "graph_nodes": len(query_app.get_graph().nodes),
    }
    # 依赖不健康时返回 503，前端与编排层都能直接感知
    return JSONResponse(status_code=200 if ok else 503, content=payload)


# ===================== 启动入口 =====================
if __name__ == "__main__":
    # 8000 被 Milvus Attu 占用、8010 被导入服务占用，查询服务使用 8011
    logger.info("Query Service 服务启动中...")
    uvicorn.run(
        app,  # 位置传参，避免与包名 app 产生歧义
        host=os.getenv("APP_HOST", "127.0.0.1"),  # docker 内通过 APP_HOST=0.0.0.0 对外暴露
        port=int(os.getenv("QUERY_PORT", "8011")),  # 8011 被系统保留时可用 QUERY_PORT 覆盖
    )