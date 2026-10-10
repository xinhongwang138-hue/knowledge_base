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
from typing import Any, Dict, List

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


async def mcp_search(query: str) -> List[Dict[str, Any]]:
    """
    异步：调用百炼 MCP 联网搜索

    :param query: 搜索关键词
    :return: 解析并清洗后的 [{title, url, snippet}, ...]
    :raise Exception: 连接或调用失败时抛出，由调用方降级处理
    """
    from agents.mcp import MCPServerStreamableHttp

    if not mcp_config.is_ready:
        raise RuntimeError(
            "百炼 MCP 未就绪：请在 .env 配置 MCP_DASHSCOPE_BASE_URL_STREAMABLE 与 API 密钥"
        )

    # Streamable HTTP 方式；terminate_on_close=True 确保会话干净关闭
    server = MCPServerStreamableHttp(
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

    try:
        await server.connect()
        logger.info("MCP 连接已建立")
        result = await server.call_tool(
            tool_name=BAILIAN_SEARCH_TOOL,
            arguments={"query": query, "count": SEARCH_COUNT},
        )
        return _parse_mcp_result(result)
    finally:
        # 无论成功失败都要清理，否则连接泄漏
        try:
            await server.cleanup()
            logger.info("MCP 连接已清理")
        except Exception as exc:
            logger.warning(f"MCP 清理异常（可忽略）：{exc}")


def _parse_mcp_result(result) -> List[Dict[str, Any]]:
    """
    解析 MCP 返回结果

    MCP 原始结构（content 里套了一层 JSON 字符串）：
        {"type": "tool_call", "content": [{"text": "{\\"pages\\": [...]}"}]}

    解析步骤：
        1. 只处理 type == "tool_call" 的响应
        2. 取出 content[0].text 并json.loads 成对象
        3. 遍历 pages，只保留 title / url / snippet
        4. 全字段去空格，剔除 snippet 为空的无效结果

    :param result: MCP call_tool 的返回值
    :return: 清洗后的文档列表
    """
    docs: List[Dict[str, Any]] = []

    # 兼容 dict 与对象两种返回形态
    payload = result
    if not isinstance(payload, dict):
        payload = getattr(result, "__dict__", {}) or {}

    if payload.get("type") != "tool_call":
        logger.warning(f"MCP 返回非 tool_call 类型: {payload.get('type')}")
        return docs

    contents = payload.get("content") or []
    if not contents:
        return docs

    text = contents[0].get("text") if isinstance(contents[0], dict) else None
    if not text:
        return docs

    # text 本身是 JSON 字符串（注意里面可能又套了一层）
    inner = json.loads(text)
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
        # 同步-异步桥接：LangGraph 节点是同步的，MCP SDK 是异步的
        docs = asyncio.run(mcp_search(query))
        logger.info(f"联网搜索完成，获得 {len(docs)} 条结果")
    except Exception as exc:
        logger.error(f"联网搜索失败，降级为空结果：{exc}")
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