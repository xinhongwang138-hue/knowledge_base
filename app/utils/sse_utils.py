"""
SSE（Server-Sent Events）工具：服务端事件推送

Web 服务把检索过程与答案推给前端时，需要一个「会话 -> 事件队列」的桥接：
    检索链路在后台跑图，通过 push_to_session 往会话队列里塞事件；
    Web 层为每个前端连接创建一个 SSE 长连接，从同一个队列里取出事件逐条推给浏览器。

事件类型（SSEEvent）：
    DELTA —— 增量文本（打字机效果的核心）
    FINAL —— 最终完整结果（答案 + 图片 + 状态）
    ERROR —— 出错信息

为什么用队列而不是直接返回 StreamingResponse：
    检索是后台任务、推送是 HTTP 响应，两者是不同执行流，
    需要一个中间缓冲把「生产的快」与「消费的慢」解耦。
"""
import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from app.core.logger import logger


class SSEEvent(str, Enum):
    """SSE 事件类型枚举"""

    READY = "ready"  # 连接握手成功
    PROGRESS = "progress"  # 节点执行进度（done_list / running_list）
    DELTA = "delta"  # 增量文本（打字机效果）
    FINAL = "final"  # 最终结果
    ERROR = "error"  # 错误


@dataclass
class Event:
    """
    单条 SSE 事件

    :param event: 事件类型（SSEEvent）
    :param data: 事件负载（会被序列化成 JSON 作为数据行）
    """

    event: SSEEvent
    data: Dict[str, Any] = field(default_factory=dict)


# session_id -> asyncio.Queue[Event]
# 注意：本字典在单进程内有效。若用 uvicorn --workers N 多进程启动，
# 各进程内存独立，SSE 推送会失效，需改用 Redis Pub/Sub 等外部通道。
_SESSIONS: Dict[str, "asyncio.Queue[Event]"] = {}


def get_session_queue(session_id: str, maxsize: int = 1000) -> "asyncio.Queue[Event]":
    """
    获取（必要时创建）某个会话的事件队列

    :param session_id: 会话唯一标识
    :param maxsize: 队列最大长度，满了丢弃最旧事件，防止慢客户端把内存撑爆
    :return: 该会话的事件队列
    """
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = asyncio.Queue(maxsize=maxsize)
    return _SESSIONS[session_id]


def push_to_session(session_id: str, event: SSEEvent, data: Optional[Dict[str, Any]] = None) -> bool:
    """
    向指定会话推送一条事件（同步接口，可在图的节点逻辑里直接调用）

    :param session_id: 会话唯一标识
    :param event: 事件类型
    :param data: 事件负载
    :return: 是否投递成功（队列满时丢弃最旧事件后返回 True）
    """
    if not session_id:
        logger.warning("push_to_session 收到空 session_id，事件被丢弃")
        return False

    q = _SESSIONS.get(session_id)
    if q is None:
        # 还没有客户端连上来监听，先记一条警告；不排队避免内存泄漏
        logger.debug(f"会话 {session_id} 暂无 SSE 监听者，事件 {event} 未排队")
        return False

    payload = Event(event=event, data=data or {})
    try:
        q.put_nowait(payload)
        return True
    except asyncio.QueueFull:
        # 队列满：丢弃最旧的一条再放入新事件，保证「最新状态」总能送达
        try:
            q.get_nowait()
            q.put_nowait(payload)
            logger.warning(f"会话 {session_id} SSE 队列已满，丢弃最旧事件后重入")
            return True
        except Exception as exc:
            logger.warning(f"会话 {session_id} SSE 投递失败：{exc}")
            return False


def format_sse(event: Event) -> str:
    """
    把事件对象格式化为 SSE 协议文本

    格式约定：
        event: <事件名>
        data: <JSON负载>

        <空行>            <- 必须，事件分隔符

    :param event: 事件对象
    :return: 可直接推给浏览器的 SSE 文本
    """
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event.value}\ndata: {payload}\n\n"


def create_sse_queue(session_id: str, maxsize: int = 1000) -> "asyncio.Queue[Event]":
    """
    创建（并确保存在）某个会话的事件队列

    这是 get_session_queue 的语义化别名：Web 层在提交任务时用「创建」语义更直观。
    内部行为与 get_session_queue 完全一致 —— 已存在则复用，避免重置正在消费的队列。

    :param session_id: 会话唯一标识
    :param maxsize: 队列最大长度
    :return: 该会话的事件队列
    """
    return get_session_queue(session_id, maxsize)


def push_progress(session_id: str, done_list=None, running_list=None, failed_list=None) -> bool:
    """
    推送节点执行进度事件（前端用它刷新进度条）

    :param session_id: 会话唯一标识
    :param done_list: 已完成节点列表
    :param running_list: 运行中节点列表
    :param failed_list: 失败节点列表
    :return: 是否投递成功
    """
    return push_to_session(
        session_id,
        SSEEvent.PROGRESS,
        {
            "done_list": list(done_list or []),
            "running_list": list(running_list or []),
            "failed_list": list(failed_list or []),
        },
    )


def close_session(session_id: str) -> None:
    """
    关闭会话并清理其队列（防止长期运行时内存无限增长）

    :param session_id: 会话唯一标识
    """
    _SESSIONS.pop(session_id, None)
    logger.debug(f"已关闭 SSE 会话：{session_id}")


async def sse_generator(session_id: str, request=None, keepalive: float = 15.0):
    """
    SSE 事件生成器（异步生成器）

    在 FastAPI 里用 StreamingResponse(media_type="text/event-stream") 包装本函数，
    即可为每个前端连接持续推送该会话的事件。

    先发一条 READY 事件做握手确认，让前端知道「管道已连通」，
    而不是干等下一条业务数据（否则前端无法区分「还在算」和「连上了」）。

    :param session_id: 会话唯一标识
    :param request: FastAPI 的 Request 对象（可选）。传入后可以检测客户端断开，
                    及时结束生成器、释放资源，避免客户端关闭页面后后台还在空推
    :param keepalive: 心跳间隔（秒），期间无事件则发注释行防代理断连
    """
    # 握手确认：告诉前端连接已建立
    yield format_sse(Event(event=SSEEvent.READY, data={"session_id": session_id}))

    q = get_session_queue(session_id)
    while True:
        # 客户端已断开则结束，避免无意义的空转
        if request is not None and await request.is_disconnected():
            logger.info(f"会话 {session_id} 的客户端已断开，结束 SSE 推送")
            break
        try:
            # 等待新事件；超时则发心跳
            evt = await asyncio.wait_for(q.get(), timeout=keepalive)
            yield format_sse(evt)
            # 收到 FINAL 后结束本次连接
            if evt.event == SSEEvent.FINAL:
                break
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"


if __name__ == "__main__":
    # 自测：推两条 DELTA + 一条 FINAL，再取出验证顺序与格式
    sid = "test_sse_001"
    get_session_queue(sid)
    print("DELTA 投递:", push_to_session(sid, SSEEvent.DELTA, {"delta": "你"}))
    print("DELTA 投递:", push_to_session(sid, SSEEvent.DELTA, {"delta": "好"}))
    print("FINAL 投递:", push_to_session(sid, SSEEvent.FINAL, {"answer": "你好", "status": "completed"}))

    q = get_session_queue(sid)
    while not q.empty():
        evt = q.get_nowait()
        print(f"取出事件: {format_sse(evt)!r}")
    close_session(sid)
    print("SSE 工具自测结束")