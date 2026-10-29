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
import os
import re
import sys
from typing import Any, Dict, List

from app.core.logger import logger, node_log, step_log
from app.lm.reranker_utils import rerank_scores
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# ---- 精排候选上限 ----
# Cross-Encoder 要把「问题 + 文档」拼成一对逐条前向，CPU 上每条约 0.5~1s，
# 候选数直接决定精排耗时（这也是单问 100s+ 的主要构成之一）。
# RRF 已经给出粗排顺序，取前 N 条足以覆盖真正相关的片段，无需全量精排。
RERANK_MAX_CANDIDATES = int(os.environ.get("RERANK_MAX_CANDIDATES", "20"))

# ---- 动态 TopK 参数 ----
# 硬上限：最多取前 N 条
RERANK_MAX_TOPK = 10
# 硬下限：至少保留前 N 条
# 取 3 而不是 1：精排分是「这一段与问题有多像」，不是「这一段够不够回答」。
# 措辞和问题高度重合的切片（重复文档的近似副本、只有图片说明的切片）很容易拿到
# 0.97+ 的分数，一旦下限是 1，断崖就把其余全部截掉——实测出现过「只返回 1 条上下文、
# 而答案在另一条里」的情况。保留 3 条的成本只有几百字上下文，却能兜住这种单点失真。
RERANK_MIN_TOPK = 3
# 断崖阈值（相对）：相邻分数跌幅超过 25% 视为断崖
RERANK_GAP_RATIO = 0.25
# 断崖阈值（绝对）：相邻分数跌幅超过 0.5 视为断崖
RERANK_GAP_ABS = 0.5

# 无信息量的章节名：整串只由数字 / 下划线 / 标点组成（源文档的空占位标题）
SECTION_TITLE_NOISE = re.compile(r"[\d\W_]+", re.UNICODE)


def _section_title(entity: Dict[str, Any]) -> str:
    """取切片所属章节名：去掉 Markdown 的 # 前缀与多余空白

    形如 `_1` / `_2` / `3` 的标题是源文档里的空占位标题，对精排没有任何信息量，
    补进头部只会给每一片都加上同样的噪声，这里直接当没有。
    """
    for key in ("parent_title", "title"):
        raw = (entity.get(key) or "").strip()
        if not raw:
            continue
        name = raw.lstrip("#").strip()
        if name and not SECTION_TITLE_NOISE.fullmatch(name):
            return name
    return ""


def build_rerank_text(entity: Dict[str, Any], content: str) -> str:
    """
    精排输入 = 「产品名｜所属章节」+ 正文

    为什么不能直接把正文丢给 Cross-Encoder：
        手册类切片大量使用「本设备」「此插头」「该有害物质」这类指代，正文里既没有
        产品名也没有章节名，精排无从判断这一段到底在讲什么，只能按行文风格打分。
        实测（HAK180 手册，问「电源插头有什么特殊要求」）：
            · 「本设备装有接地的插头…」——答案原文，只有 0.09；
            · 「![设备电源线安全须知：仅限 AC 220-240V…]」——图注，反而是 0.99。
        图注是完整自述句，指代残缺的要点列表不是，分数于是整体倒挂。

    产品名与章节名各补一半，实测差异（两条失败用例的目标切片位次，6 表示落在 Top3 之外）：
        裸正文 6/6、只补章节名 6/6、只补产品名 2/2、产品名+章节名 2/1。
        产品名才是关键那一档（解决「这是谁」），章节名再加一点精度（解决「讲哪一节」）：
        「产品中有害物质的名称及含量」补上后，才和「产品有害物质表」的问法对得上
        （该切片精排分 0.03 → 0.99）。只补章节名完全无效——`# 电源线` 帮不到「电源插头」。

    注意检索侧本来就没有这个问题：入库时向量是用 f"商品：{item_name}，介绍：{content}"
    生成的（见 node_bge_embedding / scripts/reindex_chunks.py）。真正不齐的是精排：
    它吃的是裸正文，与向量侧看到的文本不一致。这里补齐，让两侧对齐。

    :param entity: 切片实体（含 item_name / parent_title）
    :param content: 切片正文
    :return: 用于精排的文本；取不到产品与章节时退化为原正文
    """
    item_name = (entity.get("item_name") or "").strip()
    section = _section_title(entity)
    header = "｜".join(x for x in (item_name, section) if x)
    return f"{header}\n{content}" if header else content


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> dict:
    """
    Rerank 节点：合并 → 打分 → 动态截断

    :param state: 含 rrf_chunks / web_search_docs / rewritten_query
    :return: {"reranked_docs": [...], "rerank_candidates": [...]}
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
    # rerank_candidates 是截断前的完整打分结果：图片按「同文档亲和」取用时不受断崖截断影响，
    # 详见 node_answer_output._extract_images_for_answer_docs
    return {"reranked_docs": topk_docs, "rerank_candidates": scored_docs}


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
                # 精排专用文本：补上「产品名｜章节名」，把「本设备」这类指代补全。
                # 不直接改 text，是因为 text 还要原样进 Prompt 与出处摘要，
                # 多加一行头会污染引用展示。
                "rerank_text": build_rerank_text(entity, content),
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "title": entity.get("title") or entity.get("item_name") or "",
                # 章节归属：标题是切片的首个标题行，parent_title 才是它所属的那一节。
                # 出处要能说清"哪份文档的哪一段"，两者缺一都会让引用落到模糊位置。
                "parent_title": (entity.get("parent_title") or "").strip(),
                # 文档级归属键：答案节点按「同文档亲和」取图时要用，
                # 注意这里的 doc_id 实际是 chunk 级 id，做不了文档级归属
                "file_title": (entity.get("file_title") or "").strip(),
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
                # 联网结果没有产品/章节归属，精排就用原文
                "rerank_text": text,
                "doc_id": None,  # 联网结果无固定 ID
                "chunk_id": None,
                "title": (doc.get("title") or "").strip(),
                "url": (doc.get("url") or "").strip(),
                "source": "web",
            }
        )

    logger.info(f"step_1: 合并完成，共 {len(doc_items)} 条标准文档")
    # 跨源去重：同一篇内容可能既被本地切片命中、又被联网搜到，重复占位会挤掉真实候选，
    # 也会让答案来源区出现两条一样的引用。本地优先保留。
    doc_items = dedupe_docs(doc_items)
    # 候选上限：避免为几十条明显不相关的片段付精排成本（RRF 顺序即粗排可信度）
    if len(doc_items) > RERANK_MAX_CANDIDATES:
        logger.info(
            f"step_1: 候选超上限，按粗排顺序截断 {len(doc_items)} → {RERANK_MAX_CANDIDATES} 条"
        )
        doc_items = doc_items[:RERANK_MAX_CANDIDATES]
    return doc_items


# 正文指纹长度：只保留字母数字与汉字，截断到该长度再比较
DEDUP_FINGERPRINT_LEN = 150


def _fingerprint(text: str) -> str:
    """正文指纹：去掉标点空白后截断，用于判断两段文本是否讲同一件事"""
    return "".join(ch for ch in (text or "") if ch.isalnum())[:DEDUP_FINGERPRINT_LEN]


def dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按正文指纹去重，保持原有顺序（本地文档排在联网文档之前，因此天然本地优先）

    :param docs: 标准文档列表
    :return: 去重后的列表
    """
    seen = set()
    kept: List[Dict[str, Any]] = []
    for doc in docs:
        fp = _fingerprint(doc.get("text") or "")
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        kept.append(doc)
    if len(kept) != len(docs):
        logger.info(f"step_1: 跨源去重，{len(docs)} 条 → {len(kept)} 条")
    return kept


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
    # 用 rerank_text（带「产品名｜章节名」）而不是裸正文，原因见 build_rerank_text
    texts = [d.get("rerank_text") or d["text"] for d in doc_items]

    # 内存腾挪已内聚到 rerank_scores（它内部会先释放 BGE-M3 再加载/前向，
    # 整段在同一把锁内完成），这里不再重复释放，避免职责分散。
    try:
        scores = rerank_scores(question, texts)
    except Exception as exc:
        # 模型加载/推理失败不该让整条链路断在这里，降级为保持原顺序
        logger.error(f"step_2: 重排序失败，降级为原始顺序：{exc}")
        return [
            {**{k: v for k, v in item.items() if k != "rerank_text"}, "score": 0.0}
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
    # rerank_text 只服务于打分，打完即弃：
    # 它和 text 内容重复，留在 state 里会白占内存，也会被历史记录/调试输出带上
    for doc in scored_docs:
        doc.pop("rerank_text", None)
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