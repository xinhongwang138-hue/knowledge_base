"""
节点：Rerank 精排（node_rerank）

职责：RRF 只能融合**同源**（都是向量库切片）的结果，
     面对本地切片 + 联网网页这种**异源**数据就力不从心了。
     Rerank 用Cross-Encoder 模型把「问题 + 文档」一起读一遍，
     像阅读理解一样判断相关性，精度高但慢，所以只用于初筛后的小批量精排。

三步：
    1. 合并文档：本地 RRF 切片 + 联网搜索结果，统一成标准格式
    2. 精确打分：BGE Reranker 对每对「问题-文档」打分
    3. 动态 TopK：按得分的「断崖」智能截断，而不是机械取前 N 条

为什么用动态 TopK：
    固定取 Top5 会把低分文档硬塞进上下文，干扰 LLM 生成。
    动态截断看分数分布：若前几名（如 95/93/89）明显领先后续（77/71），
    说明后面是凑数的，取前三即可。
"""
import sys
from typing import Any, Dict, List

from app.core.logger import logger, node_log, step_log
from app.lm.reranker_utils import rerank_scores
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# ---- 动态 TopK 参数 ----
# 硬上限：最多取前 N 条
RERANK_MAX_TOPK = 10
# 硬下限：至少保留前 N 条
RERANK_MIN_TOPK = 1
# 断崖阈值（相对）：相邻分数跌幅超过 25% 视为断崖
RERANK_GAP_RATIO = 0.25
# 断崖阈值（绝对）：相邻分数跌幅超过 0.5 视为断崖
RERANK_GAP_ABS = 0.5


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> dict:
    """
    Rerank 节点：合并 → 打分 → 动态截断

    :param state: 含 rrf_chunks / web_search_docs / rewritten_query
    :return: {"reranked_docs": [...]}
    """
    logger.info("---Rerank 开始处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    # 1. 合并异源文档
    doc_items = step_1_merge_docs(state)
    # 2. Cross-Encoder 打分排序
    scored_docs = step_2_rerank_docs(state, doc_items)
    # 3. 动态 TopK 截断
    topk_docs = step_3_topk(scored_docs)

    add_done_task(session_id, sys._getframe().f_code.co_name)
    return {"reranked_docs": topk_docs}


@step_log("step_1_merge_docs")
def step_1_merge_docs(state: QueryGraphState) -> List[Dict[str, Any]]:
    """
    阶段一：把本地切片与联网结果合并成统一格式

    合并后的每项包含：
        text   : 用于精排的正文（本地取 content，联网取 snippet）
        title  : 标题（增强语义 / 后续引用）
        url    : 来源链接（联网有，本地为空）
        source : 来源标记 local / web

    :param state: 图状态
    :return: 标准化文档列表
    """
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []
    logger.info(f"step_1: 合并文档 - 本地RRF {len(rrf_docs)} 条, 联网 {len(web_docs)} 条")

    doc_items: List[Dict[str, Any]] = []

    # ---- 本地知识库切片 ----
    for i, doc in enumerate(rrf_docs):
        # 防御式：上游可能给的是 {"entity": {...}} 也可能直接是扁平 dict
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc
        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常 (index={i})")
            continue

        content = entity.get("content") or entity.get("text") or ""
        if not content:
            continue

        doc_id = entity.get("chunk_id") or entity.get("id")
        doc_items.append(
            {
                "text": content,
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "title": entity.get("title") or entity.get("item_name") or "",
                "url": "",
                "source": "local",
            }
        )

    # ---- 联网搜索结果 ----
    for doc in web_docs:
        if not isinstance(doc, dict):
            continue
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        if not text:
            continue
        doc_items.append(
            {
                "text": text,
                "doc_id": None,  # 联网结果无固定 ID
                "chunk_id": None,
                "title": (doc.get("title") or "").strip(),
                "url": (doc.get("url") or "").strip(),
                "source": "web",
            }
        )

    logger.info(f"step_1: 合并完成，共 {len(doc_items)} 条标准文档")
    return doc_items


@step_log("step_2_rerank_docs")
def step_2_rerank_docs(state: QueryGraphState, doc_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    阶段二：用 Cross-Encoder 对每对「问题-文档」精确打分

    :param state: 图状态（取问题文本）
    :param doc_items: 阶段一产出的标准文档
    :return: 按 score 降序排列的文档列表；打分失败时降级为原始顺序（score 置 0）
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""
    if not doc_items or not question:
        logger.warning("step_2: 跳过重排序（无文档或无问题）")
        return []

    logger.info(f"step_2: 开始重排序，待排序 {len(doc_items)} 条")
    texts = [d["text"] for d in doc_items]

    # 内存腾挪已内聚到 rerank_scores（它内部会先释放 BGE-M3 再加载/前向，
    # 整段在同一把锁内完成），这里不再重复释放，避免职责分散。
    try:
        scores = rerank_scores(question, texts)
    except Exception as exc:
        # 模型加载/推理失败不该让整条链路断在这里，降级为保持原顺序
        logger.error(f"step_2: 重排序失败，降级为原始顺序：{exc}")
        return [
            {**item, "score": 0.0}
            for item in doc_items
        ]

    scored_docs = [
        {**item, "score": score}
        for item, score in zip(doc_items, scores)
    ]
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    logger.info(
        f"step_2: 重排序完成，Top1 score={scored_docs[0]['score']:.4f}"
    )
    return scored_docs


@step_log("step_3_topk")
def step_3_topk(scored_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    阶段三：动态 TopK —— 按分数断崖截断

    从第 min_topk 名之后开始检测相邻分数落差，
    一旦超过阈值就说明后面是凑数的，立刻截断。

    例：分数 95/93/89/77/71，第3→4 名跌幅 (89-77)/89=13.5% < 25%，
        但绝对跌幅 12 > 0.5 → 触发断崖，只取前 3。

    :param scored_docs: 已按 score 降序的文档
    :return: 截断后的列表（数量 <= RERANK_MAX_TOPK）
    """
    if not scored_docs:
        return []

    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK
    topk = max_topk

    # 注意：遍历从 min_topk - 1 开始 —— 前 min_topk 名是强制保留的，不参与断崖判断
    if topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score") or 0.0
            s2 = scored_docs[i + 1].get("score") or 0.0
            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)  # 加极小值防除零
            if gap >= RERANK_GAP_ABS or rel >= RERANK_GAP_RATIO:
                logger.info(
                    f"step_3: 触发断崖截断 @ index={i} "
                    f"(score {s1:.4f} -> {s2:.4f}, gap={gap:.4f}, rel={rel:.2%})"
                )
                topk = i + 1
                break

    topk_docs = scored_docs[:topk]
    logger.info(f"step_3: 截断完成，保留前 {len(topk_docs)} 条 (TopK={topk})")
    return topk_docs


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 模拟数据：前两条与问题相关，第三条是无关内容
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "HAK 180 烫金机怎么操作？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "content": "HAK 180 烫金机的操作面板位于机器正前方，开启电源后先设置温度", "title": "操作面板", "item_name": "HAK180烫金机"},
            {"chunk_id": "local_2", "content": "进行局部烫金时调节侧面旋钮，选择区域后保存生效", "title": "局部烫金", "item_name": "HAK180烫金机"},
            {"chunk_id": "local_3", "content": "本产品采用环保材料，符合 RoHS 认证要求", "title": "环保声明", "item_name": "HAK180烫金机"},
        ],
        "web_search_docs": [
            {"title": "烫金机使用教程", "url": "http://example.com/1", "snippet": "烫金机操作步骤详解，建议温度 110℃"},
        ],
        "is_stream": False,
    }

    try:
        result = node_rerank(dict(mock_state))
        reranked = result.get("reranked_docs", [])
        print(f"\n输入文档:本地 {len(mock_state['rrf_chunks'])} + 联网 {len(mock_state['web_search_docs'])}")
        print(f"输出文档: {len(reranked)}\n最终排名:")
        for i, doc in enumerate(reranked, 1):
            preview = (doc.get("text") or "")[:30]
            print(f"  Rank{i}: [{doc.get('source'):5}] score={doc.get('score'):>9.4f} {preview}")
        print("\n期望：操作面板、局部烫金等烫金机相关内容排在前面；"
              "「RoHS 认证」这类无关内容得分最低")
    except Exception as exc:
        logger.exception(f"测试运行期间发生未捕获异常: {exc}")