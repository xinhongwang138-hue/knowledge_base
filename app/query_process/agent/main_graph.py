"""
检索主图（Query Graph）

采用 Top-Down 开发模式：先搭骨架、串主图、验证流程通畅，再逐个填充业务逻辑。

链路结构：
    用户查询入口 → node_item_name_confirm
                      ├─(state 已有 answer：反问或拒绝回答)→ node_answer_output → END
                      ↓
        ┌─────────────┼─────────────┐   三路并发
  node_search_embedding  node_search_embedding_hyde  node_web_search_mcp
        └─────────────┼─────────────┘
                      ↓
                 node_rrf（融合粗排）
                      ↓
                 node_rerank（Cross-Encoder 精排）
                      ↓
                 node_answer_output（生成答案）→ END

【重要】三个检索节点必须各自只返回自己负责的字段，绝不能 return state：
并行分支同时写同一 state 字段会触发 LangGraph 的
InvalidUpdateError: Can receive only one value per step。
"""
from langgraph.graph import END, StateGraph

# 【必须放在最前】限制 MKL 线程数，早于任何 torch 初始化（向量检索会用到 BGE-M3）
import app.core.torch_env as _torch_env  # noqa: F401  导入即生效

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

# ================ 1. 初始化状态图 ================
builder = StateGraph(QueryGraphState)

# ================ 2. 注册节点 ================
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)

# ================ 3. 设置入口 ================
builder.set_entry_point("node_item_name_confirm")


# ================ 4. 条件路由：主体确认后分流 ================
def route_after_item_confirm(state: QueryGraphState):
    """
    主体确认节点之后的路径分流

    返回 state 中已有 answer 时直接进入答案输出节点，跳过全部检索。
    这主要发生在 node_item_name_confirm 无法确定唯一商品型号的场景：

        1. 多选一（反问用户）：
           场景：用户问得太模糊（如「华为P60」），库里有「华为P60 128G」和「华为P60 Art」
                 两个型号，置信度都不足以直接确认
           处理：生成反问句作为 answer，如「您是想问以下哪个产品：华为P60 128G、华为P60 Art？」
           结果：把这句话发给用户让他选，不再检索文档

    新增逻辑：
        2. 查无此人但走通用模式（has_product=False）：
           场景：未提取到商品名，标记为通用问题
           处理：跳过硬编码拒识，直接进入检索节点
           结果：走全库检索 + LLM 生成通用回答

    :param state: 当前图状态
    :return: 目标节点名；正常情况返回三个检索节点名构成元组，表示并发执行
    """
    if state.get("answer"):
        # 有前置答案（反问），直接输出，不再检索
        return "node_answer_output"
    # 正常路径：三路并发检索（含 has_product=False 的通用模式）
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


# 条件边必须显式提供路径映射表，否则图是不完整的
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
    },
)

# ================ 5. 三路检索 → 汇合到 RRF ================
# LangGraph 会等三个节点都执行完才进入下一步，天然形成「同步栅栏」
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")

# ================ 6. 后续固定流转 ================
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# ================ 7. 编译 ================
query_app = builder.compile()


if __name__ == "__main__":
    """
    主图连通性测试（骨架阶段验证）

    Top-Down 模式的第3 步：在填充业务逻辑前，先确认图能跑通、路径正确。
    """
    import json

    from app.core.logger import logger
    from app.query_process.agent.state import create_query_default_state

    logger.info("===== 开始测试检索主图 =====")

    # 场景1：正常路径（三路并发 -> RRF -> Rerank -> 答案输出）
    initial_state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
    )
    final_state = None

    # stream_mode="updates" 只返回本节点更新的增量，便于看清「哪个节点执行了」
    for event in query_app.stream(initial_state, stream_mode="updates"):
        for node_name in event.keys():
            logger.info(f"节点执行完成: {node_name}")
            final_state = event

    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False, default=str)}")
    logger.info("图结构:")
    query_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")