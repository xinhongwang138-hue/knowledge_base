"""
节点：联网兜底（node_web_fallback）

职责：只有当**本地知识库检索结果不足**时，才按需联网补充资料，并把联网片段
     并入精排候选，重新输出最终 Top-K。

为什么改成「按需」：
    原先是三路并发（稠密 + HyDE + 联网）无条件同时跑，导致：
        1. 每个问题都要付联网的延迟（实测单问 100s+ 的主因之一）；
        2. 库里本来有答案时，联网结果与本地片段说的是同一篇内容，来源区出现重复条目；
        3. 联网内容与产品手册混排，答案口径容易被外部网页带偏。
    现在由 node_rerank 之后的 route_after_rerank 判定：本地 Top1 分数达标就直接作答，
    不达标才走本节点。

为什么只给联网片段单独打分：
    本地片段在 node_rerank 里已经打过一次分（rerank_candidates），
    这里再全量重排等于把 2GB 模型的前向重跑一遍，纯属浪费；
    只对新增的几条联网片段打分，与已有分数合并排序即可，口径完全一致。

【重要】本节点只在兜底路径执行，正常路径不会触发；返回字段不与其它节点冲突。
"""
import os
import sys
from typing import Any, Dict, List, Optional

from app.core.logger import logger, node_log
from app.lm.reranker_utils import rerank_scores
from app.query_process.agent.nodes.node_rerank import _fingerprint, step_3_topk
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def _build_web_docs(web_docs: List[Dict[str, Any]], seen: set) -> List[Dict[str, Any]]:
    """
    把 MCP 联网结果转成与 node_rerank 一致的标准文档，并剔除与本地重复的条目

    :param web_docs: node_web_search_mcp 返回的原始联网结果
    :param seen: 已存在内容的指纹集合（原地更新）
    :return: 标准化后的联网文档列表
    """
    built: List[Dict[str, Any]] = []
    for doc in web_docs or []:
        if not isinstance(doc, dict):
            continue
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        if not text:
            continue
        fp = _fingerprint(text)
        if fp and fp in seen:
            logger.info("step_1: 联网片段与已有内容重复，跳过")
            continue
        if fp:
            seen.add(fp)
        built.append(
            {
                "text": text,
                "doc_id": None,
                "chunk_id": None,
                "title": (doc.get("title") or "").strip(),
                "file_title": "",
                "url": (doc.get("url") or "").strip(),
                "source": "web",
            }
        )
    return built


@node_log("node_web_fallback")
def node_web_fallback(state: QueryGraphState) -> dict:
    """
    联网兜底：联网 → 只给联网片段打分 → 与本地已打分结果合并 → 重新断崖截断

    :param state: 含 rerank_candidates / reranked_docs / rewritten_query
    :return: {"web_search_docs", "reranked_docs", "rerank_candidates",
              "web_triggered", "web_trigger_reason"}
    """
    logger.info("---node_web_fallback: 本地结果不足，触发联网兜底---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    local_scored: List[Dict[str, Any]] = list(
        state.get("rerank_candidates") or state.get("reranked_docs") or []
    )

    # 1. 调联网检索（复用原节点，未配置密钥时返回空列表，不影响主链路）
    web_result = node_web_search_mcp(state) or {}
    raw_web_docs = web_result.get("web_search_docs") or []

    # 2. 跨源去重：本地优先，联网内容与本地重复的直接丢弃
    seen = {_fingerprint(d.get("text") or "") for d in local_scored}
    seen.discard("")
    web_docs = _build_web_docs(raw_web_docs, seen)

    reason = (
        should_trigger_web_fallback(state)
        or state.get("web_trigger_reason")
        or "本地命中不足"
    )
    if not web_docs:
        logger.warning(
            f"node_web_fallback: 联网未补充到新内容（原始 {len(raw_web_docs)} 条），"
            f"保持本地结果"
        )
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {
            "web_search_docs": [],
            "web_triggered": True,
            "web_trigger_reason": reason,
        }

    # 3. 只对联网片段打分（本地分数已存在，不重复计算）
    question = state.get("rewritten_query") or state.get("original_query") or ""
    try:
        web_scores = rerank_scores(question, [d["text"] for d in web_docs])
    except Exception as exc:
        logger.error(f"node_web_fallback: 联网片段打分失败，降级保留本地结果：{exc}")
        add_done_task(session_id, sys._getframe().f_code.co_name)
        return {
            "web_search_docs": raw_web_docs,
            "web_triggered": True,
            "web_trigger_reason": reason,
        }

    scored_web = [{**doc, "score": score} for doc, score in zip(web_docs, web_scores)]

    # 4. 与本地候选合并后统一排序，再走同一套动态 TopK 截断
    merged = local_scored + scored_web
    merged.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    topk_docs = step_3_topk(merged)

    logger.info(
        f"node_web_fallback: 合并完成，本地 {len(local_scored)} 条 + 联网 {len(scored_web)} 条 "
        f"→ 最终保留 {len(topk_docs)} 条（触发原因：{reason}）"
    )
    add_done_task(session_id, sys._getframe().f_code.co_name)
    return {
        "web_search_docs": raw_web_docs,
        "reranked_docs": topk_docs,
        "rerank_candidates": merged,
        "web_triggered": True,
        "web_trigger_reason": reason,
    }


def should_trigger_web_fallback(state: QueryGraphState) -> Optional[str]:
    """
    判定是否需要联网兜底（供主图条件边使用）

    判定口径：
        1. 已经兜底过 → 不再兜底（一次为限，避免来回循环）
        2. 环境变量 WEB_FALLBACK_ENABLED=0 → 不兜底
        3. 本地没有任何可用片段 → 兜底
        4. 本地 Top1 精排分数低于阈值 → 兜底（说明库里没有能回答这个问题的内容）

    :return: 触发原因为字符串（表示需要兜底）；None 表示本地结果已足够
    """
    if state.get("web_triggered"):
        return None

    if os.environ.get("WEB_FALLBACK_ENABLED", "1").lower() in ("0", "false", "no"):
        return None

    docs = state.get("reranked_docs") or []
    if not docs:
        return "本地无可用片段"

    top_score = docs[0].get("score") or 0.0
    try:
        threshold = float(os.environ.get("WEB_FALLBACK_MIN_SCORE", "0.5"))
    except ValueError:
        threshold = 0.5
    if top_score < threshold:
        return f"本地最高分 {top_score:.3f} < 阈值 {threshold}"
    return None
