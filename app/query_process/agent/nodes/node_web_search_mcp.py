"""
节点：MCP 联网搜索（node_web_search_mcp）

职责：通过 MCP (Model Context Protocol) 协议调用阿里云百炼的联网搜索服务，
补充知识库之外的实时信息。

同步-异步桥接（本节点的关键技术点）：
    OpenAI MCP SDK 的 connect() / call_tool() / cleanup() 全是 async 函数，
    而 LangGraph 节点默认同步执行。Python 语法上同步函数不能直接调异步方法，
    所以用 `asyncio.run()` 临时起一个事件循环跑完异步代码再回到同步逻辑。

MCP 三种传输方式：
    - stdio           本机子进程通信，适合本机工具/桌面插件
    - SSE             HTTP 下行流，早期 Web 方案，上下行分离、代理兼容性麻烦
    - Streamable HTTP 统一可流式传输，云上生产首选
    百炼新开通的 bailian_web_search 默认是 streamablehttp，故本实现用它。

【当前状态】密钥/地址未就绪时优雅降级：返回空列表，不影响检索主链路。

【重要】本节点与另两个检索节点三路并发，只返回自己负责的字段。
"""
import asyncio
import json
import sys
import threading
from typing import Any, Callable, Dict, List

from app.clients.bailian_mcp_config import mcp_config
from app.core.logger import logger, node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 百炼联网搜索工具的固定标识，不可改
BAILIAN_SEARCH_TOOL = "bailian_web_search"
# 期望返回的搜索结果条数（百炼建议 1-10）
SEARCH_COUNT = 5
# MCP 连接与读取超时（秒）
MCP_TIMEOUT = 300

# P1-3：模块级常驻事件循环与 MCP 连接复用。
# 背景：LangGraph 三路检索并行分支会在线程池中执行同步节点，此时若再调用
# asyncio.run() 会抛 "asyncio.run() cannot be called from a running event loop"，
# 且每次查询都 connect/cleanup 浪费大量握手开销。
# 方案：独立后台线程跑常驻事件循环，coroutine 通过 run_coroutine_threadsafe 提交；
# MCP Server 懒加载并复用，连接失效时自动重建。
_MCP_LOOP: "asyncio.AbstractEventLoop | None" = None
_MCP_LOOP_THREAD: "threading.Thread | None" = None
_MCP_LOOP_LOCK = threading.Lock()
_MCP_SERVER: Any = None
_MCP_CONNECTED: bool = False
_MCP_SERVER_LOCK = threading.Lock()


def _ensure_mcp_loop() -> "asyncio.AbstractEventLoop":
    """确保存在一个后台线程常驻事件循环（只启动一次，之后复用）"""
    global _MCP_LOOP, _MCP_LOOP_THREAD
    with _MCP_LOOP_LOCK:
        if _MCP_LOOP is None or _MCP_LOOP.is_closed():
            _MCP_LOOP = asyncio.new_event_loop()
            _MCP_LOOP_THREAD = threading.Thread(
                target=_MCP_LOOP.run_forever,
                name="mcp-search-loop",
                daemon=True,
            )
            _MCP_LOOP_THREAD.start()
    return _MCP_LOOP


def _run_coro_in_loop(coro_factory: Callable[[], "asyncio.Future"]) -> Any:
    """把 async 协程提交到常驻事件循环并同步等待结果（线程安全）"""
    loop = _ensure_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)
    return future.result(timeout=MCP_TIMEOUT + 30)


def _reset_mcp_server() -> None:
    """连接失效时重置，下次调用重新创建"""
    global _MCP_SERVER, _MCP_CONNECTED
    with _MCP_SERVER_LOCK:
        _MCP_SERVER = None
        _MCP_CONNECTED = False


def _get_mcp_server() -> Any:
    """懒加载并返回复用的 MCP Server 对象"""
    global _MCP_SERVER
    with _MCP_SERVER_LOCK:
        if _MCP_SERVER is not None:
            return _MCP_SERVER
        from agents.mcp import MCPServerStreamableHttp

        _MCP_SERVER = MCPServerStreamableHttp(
            name="bailian_search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": mcp_config.api_key},
                "timeout": MCP_TIMEOUT,
                "sse_read_timeout": MCP_TIMEOUT,
                "terminate_on_close": True,
            },
            max_retry_attempts=2,
        )
        return _MCP_SERVER


async def mcp_search(query: str) -> List[Dict[str, Any]]:
    """
    异步：调用百炼 MCP 联网搜索（复用常驻连接）

    :param query: 搜索关键词
    :return: 解析并清洗后的 [{title, url, snippet}, ...]
    :raise Exception: 连接或调用失败时抛出，由调用方降级处理
    """
    global _MCP_CONNECTED

    if not mcp_config.is_ready:
        raise RuntimeError(
            "百炼 MCP 未就绪：请在 .env 配置 MCP_DASHSCOPE_BASE_URL_STREAMABLE 与 API 密钥"
        )

    server = _get_mcp_server()
    try:
        # 连接复用：首次调用才 connect，后续直接 call_tool
        if not _MCP_CONNECTED:
            await server.connect()
            _MCP_CONNECTED = True
            logger.info("MCP 连接已建立（复用）")
        result = await server.call_tool(
            tool_name=BAILIAN_SEARCH_TOOL,
            arguments={"query": query, "count": SEARCH_COUNT},
        )
        return _parse_mcp_result(result)
    except Exception as exc:
        # 连接可能已失效，重置以便下次重建
        _reset_mcp_server()
        raise exc


def _parse_mcp_result(result) -> List[Dict[str, Any]]:
    """
    解析 MCP 返回结果

    openai-agents SDK 的 call_tool 返回的是 CallToolResult 对象：
        CallToolResult(
            content=[TextContent(type="text", text='{"pages": [...]}')],
            isError=False,
        )
    即 content 是列表，每个 block 是 TextContent（对象或 dict），其 .text 是一层
    JSON 字符串，里面才有真正的 pages 数组。CallToolResult 本身没有 "type" 字段——
    早期代码误以为顶层有 type=="tool_call"，导致正常返回也被判为“非 tool_call”而丢弃。

    解析步骤：
        1. 从 result.content 取出所有 content block（兼容对象与 dict 两种形态）
        2. 逐个取 block.text，json.loads 成对象
        3. 遍历 pages，只保留 title / url / snippet
        4. 全字段去空格，剔除 snippet 为空的无效结果

    :param result: MCP call_tool 的返回值（CallToolResult 或等价 dict）
    :return: 清洗后的文档列表
    """
    docs: List[Dict[str, Any]] = []

    # 兼容对象（有 .content 属性）与 dict（有 "content" 键）两种返回形态
    contents = None
    if hasattr(result, "content"):
        contents = result.content
    elif isinstance(result, dict):
        contents = result.get("content")

    if not contents:
        logger.warning("MCP 返回无 content，跳过解析")
        return docs

    for block in contents:
        # content block 兼容 TextContent 对象与 dict 两种形态
        if isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if not text:
            continue
        try:
            inner = json.loads(text)
        except json.JSONDecodeError:
            # 个别实现 text 直接是纯文本而非 JSON，无法解析为 pages，跳过
            logger.warning("MCP content 非 JSON，已跳过")
            continue

        pages = inner.get("pages") or []
        for item in pages:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            # snippet 是正文，没有摘要的条目对后续 rerank / 生成答案没有价值
            if not snippet:
                continue
            docs.append({"title": title, "url": url, "snippet": snippet})

    return docs


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState) -> dict:
    """
    联网搜索节点

    :param state: 含 session_id / rewritten_query
    :return: {"web_search_docs": [...]}；无结果时为空列表
    """
    logger.info("---node_web_search_mcp 处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    query = state.get("rewritten_query") or state.get("original_query") or ""
    docs: List[Dict[str, Any]] = []

    if not query:
        logger.warning("查询为空，跳过联网搜索")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"web_search_docs": []}

    # 未配置 MCP 时优雅降级：联网搜索只是补充手段，不该让整条链路失败
    if not mcp_config.is_ready:
        logger.warning(
            "百炼 MCP 未配置，跳过联网搜索（部署方式见笔记 9.4.1；"
            "配置后本节点会自动启用）"
        )
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"web_search_docs": []}

    try:
        # 同步-异步桥接（P1-3）：提交到常驻后台事件循环执行，连接复用，
        # 避免 LangGraph 并行分支中 asyncio.run() 与已有事件循环冲突
        docs = _run_coro_in_loop(lambda: mcp_search(query))
        logger.info(f"联网搜索完成，获得 {len(docs)} 条结果")
    except (Exception, asyncio.CancelledError) as exc:
        # 注意：Python 3.8+ 中 asyncio.CancelledError 是 BaseException 子类，
        # 单独的 `except Exception` 捕获不到，会导致联网搜索失败时直接中断整张图。
        # 联网搜索只是增强手段，任何异常（含取消）都必须降级为空结果，绝不影响主链路。
        reason = str(exc)
        hint = ""
        # 百炼侧常见现象：账户未开通 WebSearch MCP 服务时，SDK 抛出的异常形态不稳定——
        # 可能是 McpError("Session terminated")，也可能是 asyncio.CancelledError
        # （"Cancelled via cancel scope"，连接断开时 anyio 取消任务所致）。统一在此给出
        # 明确可执行的排查方向，避免误判为代码 bug。
        _key = (reason + " " + type(exc).__name__).lower()
        if any(k in _key for k in ("session terminated", "mcperror", "cancelled", "cancel scope", "terminat")):
            hint = "（多为百炼 WebSearch MCP 未在对应账户开通/激活，请在 DashScope 控制台开通该 MCP 服务后重试）"
        logger.error(f"联网搜索失败，降级为空结果：{type(exc).__name__}: {reason}{hint}")
        docs = []

    add_done_task(session_id, sys._getframe().f_code.co_name)
    return {"web_search_docs": docs}


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)
    print(f"MCP 配置状态: url={mcp_config.mcp_base_url or '(未配置)'}")
    print(f"密钥状态: {'已配置' if mcp_config.api_key else '(未配置)'}")
    print(f"is_ready: {mcp_config.is_ready}")

    test_state = {
        "session_id": "test_mcp_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "rewritten_query": "HAK 180 烫金机在出厂默认状态下如何设置局部烫金区域？",
        "is_stream": False,
    }

    result = node_web_search_mcp(dict(test_state))
    docs = result.get("web_search_docs", [])
    print(f"\n搜索结果数量: {len(docs)}")
    if docs:
        for i, d in enumerate(docs, 1):
            print(f"  {i}. {d['title']}")
            print(f"     {d['url']}")
            print(f"     {d['snippet'][:60]}...")
    else:
        print("（无结果：MCP 未配置或调用失败，属预期降级行为）")
    print("=" * 50)