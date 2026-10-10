"""
节点：RRF 倒数排名融合（node_rrf）

职责：把多路召回（向量检索、HyDE 检索、联网搜索）的结果合并去重，
按「各路排名倒数 × 权重」累加得分，输出统一的 Top-N 列表。

为什么 RRF 能融合不同来源：
    不同检索方式的分数不可比 —— 稠密向量的 COSINE 分数是0~1，
    稀疏向量的 IP 分数可能到十几，根本没法直接相加。
    RRF 绕开分数只看**排名**：第 1 名记 1/(k+1)，第 2 名记 1/(k+2)……
    这样任何来源的量纲都被统一成「排名倒数」，可以放心相加。

核心公式：
    score(doc) = Σ_来源  weight_来源 × 1 / (k + rank_来源(doc))

k 的作用（行业惯例取 60）：平滑排名的影响。
    没有 k 时公式退化为 1/rank，第 1 名得 1.0、第 2 名 0.5、第 3 名 0.33，
    单个来源的第 1 名会直接主导总得分，其他来源的排名几乎被忽略，
    违背了「多来源融合」的初衷。
    加上 k=60 后，前几名得分差异被大幅缩小（0.0164 / 0.0162 / 0.0160），
    必须多个来源都靠前才能拿高分。

直观理解：奖励「多面手」——在多个榜单都靠前的文档排名最高。

【重要】本节点在三个检索节点之后执行，此时并发已汇合，
可以返回 rrf_chunks 字段。
"""
import sys
from typing import Any, Dict, List, Tuple

from app.core.logger import logger, node_log, step_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# RRF 平滑常数，业界惯例 60。越大越强调「多次上榜」，越小越强调「单次排名靠前」
RRF_K = 60
# 融合后保留的最终数量
RRF_MAX_RESULTS = 10
# 各来源权重：向量检索与 HyDE 检索同权重
WEIGHT_EMBEDDING = 1.0
WEIGHT_HYDE = 1.0
WEIGHT_WEB = 0.8  # 联网搜索可信度略低，权重下调


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> dict:
    """
    RRF 倒数排名融合节点

    步骤：
        1. 从 state 取出各路召回结果
        2. 标准化为统一格式（_as_entity_list）
        3. 按权重执行 RRF 融合排序
        4. 截取 Top N 并写回 state

    :param state: 含 embedding_chunks / hyde_embedding_chunks / web_search_docs
    :return: {"rrf_chunks": [...]}
    """
    logger.info("---RRF (倒数排名融合) 开始处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    # 1. 标准化各路结果
    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))
    # 联网搜索结果本身就是 {title,url,snippet} 扁平字典，不做 entity 拆解
    web_docs = _as_web_list(state.get("web_search_docs"))

    logger.info(
        f"RRF 输入:向量 {len(embedding_chunks)} 条 / "
        f"HyDE {len(hyde_chunks)} 条 / 网络 {len(web_docs)} 条"
    )

    # 2. 配置各路权重
    source_weights = [
        (embedding_chunks, WEIGHT_EMBEDDING),
        (hyde_chunks, WEIGHT_HYDE),
        (web_docs, WEIGHT_WEB),
    ]

    # 3. 执行 RRF
    rrf_res = reciprocal_rank_fusion(
        source_weights, k=RRF_K, max_results=RRF_MAX_RESULTS
    )

    # 4. 取出文档列表
    rrf_chunks = [doc for doc, score in rrf_res]
    logger.info(f"RRF 融合完成，输出 {len(rrf_chunks)} 条")

    add_done_task(session_id, sys._getframe().f_code.co_name)
    return {"rrf_chunks": rrf_chunks}


@step_log("_as_entity_list")
def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """
    把上游检索结果统一规整为「实体字典」列表

    需要兼容三种形态（不同上游节点、不同 SDK 版本返回结构可能不同）：
        A. Milvus Hit 对象：有 .entity 与 .id 属性
        B. 嵌套字典：{"entity": {...}, "id": ..., "distance": ...}
        C. 扁平字典：{"chunk_id": ..., "content": ...}

    :param state_list: 上游节点返回的原始列表
    :return: 规整后的实体字典列表
    """
    out: List[Dict[str, Any]] = []
    for doc in state_list or []:
        if not doc:
            continue

        # 情况A：Milvus Hit 对象
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            content = doc.entity
            if hasattr(content, "to_dict"):
                final = content.to_dict()
            elif isinstance(content, dict):
                final = content.copy()
            else:
                try:
                    final = dict(content)
                except Exception:
                    continue
            # Hit 的主键补进实体，方便 RRF 做唯一键
            final.setdefault("id", doc.id)
            if hasattr(doc, "distance"):
                final["score"] = doc.distance

        # 情况B/C：字典
        elif isinstance(doc, dict):
            if isinstance(doc.get("entity"), dict):
                final = doc["entity"].copy()
                if "id" in doc and "id" not in final:
                    final["id"] = doc["id"]
                if "distance" in doc:
                    final["score"] = doc["distance"]
            else:
                final = dict(doc)
        else:
            continue

        if final:
            out.append(final)

    return out


def _as_web_list(state_list) -> List[Dict[str, Any]]:
    """
    规整联网搜索结果：把 snippet 转成统一的 text 字段

    联网结果与本地切片的字段不同（snippet vs content），
    统一成 text 后，下游 rerank/生成答案可以同样处理两类数据。

    :param state_list: [{title, url, snippet}, ...]
    :return: [{text, title, url, source: "web"}, ...]
    """
    out: List[Dict[str, Any]] = []
    for doc in state_list or []:
        if not isinstance(doc, dict):
            continue
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "title": (doc.get("title") or "").strip(),
                "url": (doc.get("url") or "").strip(),
                "source": "web",
            }
        )
    return out


def reciprocal_rank_fusion(
    source_weights: list,
    k: int = RRF_K,
    max_results: int = None,
) -> List[Tuple[Any, float]]:
    """
    通用带权重的 RRF 算法

    :param source_weights: [(来源文档列表, 权重), ...]
                          例：[([doc1, doc2], 1.0), ([doc2, doc3], 0.8)]
    :param k: RRF 平滑常数，默认 60
    :param max_results: 只返回前 N条，None 表示全部
    :return: [(文档, 融合得分), ...]，按得分降序
    """
    score_map: Dict[Any, float] = {}
    doc_map: Dict[Any, Any] = {}

    for docs, weight in source_weights:
        # rank 从 1 开始（第1名、第2名...），符合实际排序语义而非 0 索引
        for rank, item in enumerate(docs, start=1):
            if not isinstance(item, dict):
                continue
            # 唯一标识：本地切片用 chunk_id/id，联网结果没有 ID 用 url
            doc_id = item.get("chunk_id") or item.get("id") or item.get("url")
            if not doc_id:
                logger.warning(f"RRF: 缺少唯一标识的条目已跳过，keys={list(item.keys())[:5]}")
                continue

            # RRF 核心公式：权重 × 排名倒数
            score_map[doc_id] = score_map.get(doc_id, 0.0) + weight * (1.0 / (k + rank))
            # 只保留首次出现的内容（跨源去重）
            doc_map.setdefault(doc_id, item)

    merged = [(doc_map[cid], score) for cid, score in score_map.items()]
    merged.sort(key=lambda x: x[1], reverse=True)

    if max_results is not None:
        merged = merged[:max_results]
    return merged


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    # 构造两路有重叠的结果，验证「跨源去重 + 多榜靠前者胜出」
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK180烫金机"],
        "embedding_chunks": [
            {"entity": {"chunk_id": "A", "content": "文档A：面板位置", "item_name": "X"}},
            {"entity": {"chunk_id": "B", "content": "文档B：温度设置", "item_name": "X"}},
            {"entity": {"chunk_id": "C", "content": "文档C：清洁保养", "item_name": "X"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "B", "content": "文档B：温度设置", "item_name": "X"}},
            {"entity": {"chunk_id": "D", "content": "文档D：故障排除", "item_name": "X"}},
        ],
    }

    try:
        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])
        print(f"输入:向量 {len(mock_state['embedding_chunks'])} 条, HyDE {len(mock_state['hyde_embedding_chunks'])} 条")
        print(f"输出(去重后): {len(rrf_chunks)} 条")
        print("-" * 30)
        for i, doc in enumerate(rrf_chunks, 1):
            print(f"  Rank {i}: id={doc.get('chunk_id')} content={(doc.get('content') or '')[:20]}")

        # B 出现在两路且都是高分 → 应排第一；A 只在向量路排第1 → 应落后于 B
        print("\n期望：B（两路上榜）排在第一，A（仅单路）落后于 B")
    except Exception as exc:
        logger.exception(f"测试运行期间发生未捕获异常: {exc}")