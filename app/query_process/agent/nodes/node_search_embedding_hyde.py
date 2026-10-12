"""
节点：HyDE 假设性文档检索（node_search_embedding_hyde）

HyDE (Hypothetical Document Embeddings) 的思路：
    用户只给一句短问题（「烫金机怎么用」），直接拿去做向量检索，语义信号太稀疏——
    问题和知识库里的「文档」在语义空间里根本不在同一形态，匹配效果差。
    HyDE 的做法是：先让 LLM「脑补」一段可能的答案（像写了一段假想的说明书），
    这段文本更具体、更像一篇小文档，包含更多关键词与表达方式，
    再拿它去做检索，命中率往往更高。

优点：简单、通用，对短查询场景提升明显。
缺点：也可能「脑补过头」把方向带偏，所以通常与普通向量检索并行使用、
    两者结果一起交给 RRF 融合（这正是本项目三路召回的设计意图）。

本节点与另两个检索节点三路并发，只返回自己负责的字段，绝不能 return state。
"""
import sys

from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 单路召回数量（多召回给后续 RRF/Rerank 留筛选空间）
REQUEST_LIMIT = 10
# 最终返回的 Top-K
RETRIEVE_TOP_K = 5
# 稠密/稀疏权重：HyDE 文档较长、语义丰富，稠密权重可略高
RANKER_WEIGHTS = (0.8, 0.2)


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState) -> dict:
    """
    HyDE 检索节点

    执行步骤：
        1. 取改写后的问题（缺失时降级用原始问题）与已确认商品名
        2. 调 LLM 生成假设性文档（HyDE Doc）
        3. 用「问题 + 假设文档」拼接后的文本做混合检索

    :param state: 含 session_id / rewritten_query / original_query / item_names
    :return: {"hyde_embedding_chunks": [...], "hyde_doc": "假设文档"}
    """
    logger.info("---HyDE (假设文档检索) 节点开始处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    # 优先用改写后的问题，没有则退回原始问题
    query = state.get("rewritten_query") or state.get("original_query") or ""
    if not query:
        logger.error("HyDE: 未找到有效的用户查询")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    item_names = state.get("item_names") or []

    # 与向量检索节点保持一致的「无商品名」语义（P1-4）：
    # has_product=False 表示通用问题，跳过 item_name 过滤走全库检索；
    # 空 item_names 且非通用模式时同样跳过检索，避免跨产品串味。
    is_general = state.get("has_product") is False
    if not item_names and not is_general:
        logger.warning("item_names 为空且非通用模式，跳过 HyDE 检索（与向量检索节点保持一致）")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    # 阶段1：生成假设性文档
    try:
        hyde_doc = step_1_create_hyde_doc(query)
        logger.info(f"step_1: 假设文档生成成功（{len(hyde_doc)} 字符）")
    except Exception as exc:
        # HyDE 是增强手段而非必需项，失败降级为空结果，不阻断整体流程
        logger.error(f"step_1: 生成假设文档失败，降级为空结果：{exc}")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    # 阶段2：用「问题 + 假设文档」做混合检索
    try:
        res = step_2_search_embedding_hyde(query, hyde_doc, item_names)
        chunks = res[0] if res else []
        logger.info(f"step_2: HyDE 检索完成，召回 {len(chunks)} 条")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"hyde_embedding_chunks": chunks, "hyde_doc": hyde_doc}
    except Exception as exc:
        logger.error(f"step_2: HyDE 检索异常，降级为空结果：{exc}")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"hyde_embedding_chunks": [], "hyde_doc": hyde_doc}


@step_log("step_1_create_hyde_doc")
def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    阶段1：用 LLM 生成假设性文档

    :param rewritten_query: 改写后的用户问题
    :return: LLM 生成的假设性文档
    :raise ValueError: 问题为空时抛出
    """
    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")

    llm = get_llm_client()
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    response = llm.invoke(prompt)
    hyde_doc = (response.content or "").strip()

    if not hyde_doc:
        raise ValueError("LLM 返回了空的假设文档")
    logger.debug(f"假设文档预览: {hyde_doc[:80]}...")
    return hyde_doc


@step_log("step_2_search_embedding_hyde")
def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
) -> list:
    """
    阶段2：用「问题 + 假设文档」检索切片

    为什么要拼接两者：问题保留了用户真实意图（约束方向），
    假设文档提供了饱满的语义信号（提升匹配），两者结合最稳。

    :param rewritten_query: 改写后的问题
    :param hyde_doc: 假设性文档
    :param item_names: 商品名过滤范围
    :return: Milvus 混合搜索的原始结果（二维结构）
    """
    combined_text = f"{rewritten_query} {hyde_doc}".strip()

    embeddings = generate_embeddings([combined_text])
    dense_vec = embeddings["dense"][0]
    sparse_vec = embeddings["sparse"][0]

    client = get_milvus_client()
    if not client:
        logger.error("step_2: 无法连接 Milvus")
        return []

    # 与向量检索节点保持一致的过滤逻辑，避免跨产品串味
    expr = None
    if item_names:
        quoted = ", ".join(f"'{escape_milvus_string(n)}'" for n in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"step_2: 过滤条件 {expr}")
    else:
        logger.info("step_2: 无商品名过滤，全库检索")

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=REQUEST_LIMIT,
    )

    res = hybrid_search(
        client=client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=RANKER_WEIGHTS,
        norm_score=True,
        limit=RETRIEVE_TOP_K,
        output_fields=["chunk_id", "content", "title", "item_name", "part"],
    )
    return res


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK180烫金机"],
        "is_stream": False,
    }

    try:
        result = node_search_embedding_hyde(dict(mock_state))
        hyde_doc = result.get("hyde_doc", "")
        chunks = result.get("hyde_embedding_chunks", [])
        print(f"\nHyDE 文档已生成: {bool(hyde_doc)}（{len(hyde_doc)} 字符）")
        if hyde_doc:
            print(f"文档预览: {hyde_doc[:80]}...")
        print(f"召回切片: {len(chunks)} 条")
        for i, c in enumerate(chunks, 1):
            e = c.get("entity", {}) or {}
            print(f"  {i}. score={c.get('distance'):.4f} title={e.get('title')!r}")
    except Exception as exc:
        logger.exception(f"测试运行期间发生未捕获异常: {exc}")