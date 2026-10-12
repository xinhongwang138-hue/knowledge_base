"""
节点：向量检索（node_search_embedding）

职责：拿「改写后的用户问题」在 kb_chunks 中做 BGE-M3 混合检索（稠密+稀疏），
用已确认的 item_name 做过滤限定范围，召回 Top-K 最相关的切片。

为什么要 item_name 过滤：
    不加过滤时，40 条切片里可能混着其它产品的内容，语义向量再准也会串味。
    加上 `item_name in [...]` 后，检索被限定在已确认的产品范围内，
    这也是「主体确认」节点存在的价值所在。

权重选择 0.8/0.2：
    稠密向量捕捉语义（用户问「怎么操作」能匹配到「使用步骤」），
    稀疏向量捕捉字面（型号、参数名精确匹配）。
    面向自然语言问答场景，语义权重应更高，故 0.8/0.2。

【重要】本节点与另两个检索节点三路并发，只返回自己负责的字段，
绝不能 return state（会触发 LangGraph 的 InvalidUpdateError）。
"""
import sys

from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.core.logger import logger, node_log
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 最终返回的切片数
RETRIEVE_TOP_K = 5
# 底层单路召回的数量：多召回一些给后续 rerank 留筛选空间
REQUEST_LIMIT = 10
# 稠密/稀疏权重
RANKER_WEIGHTS = (0.8, 0.2)


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState) -> dict:
    """
    核心节点：基于已确认商品名 + 改写后的问题，执行 Milvus 混合检索

    :param state: 含 session_id / rewritten_query / item_names
    :return: {"embedding_chunks": [...]}；无结果时为空列表
    """
    logger.info("---向量内容检索 开始处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    query = state.get("rewritten_query") or state.get("original_query") or ""
    item_names = state.get("item_names") or []

    if not query:
        logger.warning("查询为空，跳过向量检索")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"embedding_chunks": []}

    # 没有确认商品就不检索 —— 宁可无结果，也不要返回一堆串味的切片
    # 如果 has_product=False，说明是通用问题，跳过 item_name 过滤走全库检索
    is_general = state.get("has_product") is False
    if not item_names and not is_general:
        logger.warning("item_names 为空且非通用模式，跳过检索（避免跨产品串味）")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"embedding_chunks": []}

    logger.info(f"检索入参: query={query[:50]}..., item_names={item_names}, has_product={state.get('has_product')}")

    try:
        client = get_milvus_client()
        if not client:
            logger.error("无法连接 Milvus")
            add_done_task(session_id, sys._getframe().f_code.co_name)
            return {"embedding_chunks": []}

        # 问题向量化（稠密 + 稀疏）
        embeddings = generate_embeddings([query])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]
        logger.debug(
            f"向量生成成功: dense_dim={len(dense_vec)}, sparse_features={len(sparse_vec)}"
        )

        # item_name 过滤：has_product=False 时跳过过滤，走全库检索
        expr = None
        if item_names:
            quoted = ", ".join(f"'{escape_milvus_string(n)}'" for n in item_names)
            expr = f"item_name in [{quoted}]"
            logger.info(f"过滤表达式: {expr}")
        else:
            logger.info("无 item_name 过滤，走全库检索")

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
            output_fields=["chunk_id", "content", "title", "item_name", "part", "file_title"],
        )

        # Milvus 混合搜索返回二维结构：res[0] 才是本次查询的结果集
        chunks = res[0] if res else []
        logger.info(f"检索到 {len(chunks)} 条相关切片")
        if chunks:
            top = chunks[0]
            entity = top.get("entity", {}) or {}
            logger.debug(
                f"Top1: score={top.get('distance')}, item_name={entity.get('item_name')}"
            )

        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"embedding_chunks": chunks}

    except Exception as exc:
        # 检索失败降级为空结果，不阻断 RRF（其它路还能贡献结果）
        logger.error(f"向量检索异常：{exc}", exc_info=True)
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {"embedding_chunks": []}


# ===================== 本地测试 =====================
if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "item_names": ["HAK180烫金机"],
        "is_stream": False,
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        result = node_search_embedding(dict(test_state))
        chunks = result.get("embedding_chunks", [])
        print(f">>> 检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            entity = top1.get("entity", {}) or {}
            print(f"  score       = {top1.get('distance')}")
            print(f"  item_name   = {entity.get('item_name')}")
            print(f"  title= {entity.get('title')}")
            print(f"  content 前 100 字 = {(entity.get('content') or '')[:100]}")
        else:
            print("\n>>> 未检索到结果，请检查 Milvus 数据或 item_names 是否匹配")
    except Exception as exc:
        logger.error(f"测试运行失败: {exc}", exc_info=True)