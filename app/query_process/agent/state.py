"""
检索链路的状态定义

所有节点共享同一个 QueryGraphState 对象，用于在检索流程中流转数据。
与导入链路的 ImportGraphState 相互独立 —— 导入负责「建库」，检索负责「查库」。

字段流转顺序：
    original_query -> item_names / rewritten_query   （意图识别与改写）
                   -> embedding_chunks / hyde_... / web_search_docs （多路召回）
                   -> rrf_chunks                      （RRF 融合粗排）
                   -> reranked_docs                   （Cross-Encoder 精排）
                   -> prompt / answer                 （LLM 生成答案）
"""
import copy
from typing import List

from typing_extensions import TypedDict


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    用 TypedDict 的好处：编辑器能自动补全字段名，且不产生运行时开销。
    """

    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # ---- 检索过程中的中间数据 ----
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # ---- 排序过程中的数据 ----
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # ---- 生成过程中的数据 ----
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # ---- 辅助信息 ----
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    has_product: bool  # 商品名确认结果：True=确认到商品，False=未确认（走通用检索）


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    # P2-5：默认未确认到商品（False），避免空商品名时误走精确商品检索；
    # 节点确认到商品后由上游显式置为 True
    "has_product": False,
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段

    :param **overrides: 要覆盖的字段（关键字参数解包）
    :return: 新的状态实例（深拷贝，避免污染全局默认值）
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    """
    返回一份全新的默认状态，避免全局变量污染

    :return: 深拷贝后的默认状态
    """
    return copy.deepcopy(query_graph_default_state)


# ========================
# 状态复制函数
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据

    典型用途：同一会话里基于已有状态派生一个新分支（如流式与非流式出两版），
    此时若直接改原 state 会把上一轮结果一起改掉。

    :param state: 原始状态
    :param **overrides: 要覆盖的字段
    :return: 新的状态实例
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试1：创建默认状态并覆盖若干字段
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False,
    )
    print("初始化状态：", state)

    # 测试2：复制状态并覆盖原始问题，验证原状态不被污染
    new_state = copy_query_state(state, original_query="修改后的问题")
    print("复制后的状态 original_query：", new_state["original_query"])
    print("原状态 original_query：", state["original_query"])
    assert state["original_query"] == "华为P60怎么样?", "原状态被污染了！"
    assert new_state["original_query"] == "修改后的问题"
    print("深拷贝验证通过：修改副本不影响原状态")

    # 测试3：默认状态的列表字段应各自独立（避免共享同一个 list 对象）
    s1 = get_query_default_state()
    s2 = get_query_default_state()
    s1["item_names"].append("测试商品")
    assert s2["item_names"] == [], "两次获取的默认状态共享了同一个列表！"
    print("默认值隔离验证通过：两次获取的默认状态互不影响")

    print("===== state 测试结束 =====")