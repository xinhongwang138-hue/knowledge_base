"""
节点：主体确认（node_item_name_confirm）

职责：确定用户问的是**哪个产品**，以便定位对应手册。做法是让 LLM 从问题中提取商品名，
再与知识库中已有的标准商品名做向量比对；无法确定时把候选列表返回给用户确认。

为什么需要这一层：
    用户说「华为P60」，库里可能有「华为P60 128G」「华为P60 Art」多个型号，
    直接拿去检索会召回一堆不相干的内容。这里先把模糊名称对齐到标准型号，
    后续检索就能用item_name 做精确过滤，大幅提升召回精度。

七个子步骤：
    step_1  取历史会话（供指代消解用）
    step_2  保存当前问题到历史库
    step_3  LLM 提取 item_names + 改写问题（处理「它」「这个」等代词）
    step_4  商品名向量化，在 kb_item_names 中混合检索评分
    step_5  按评分对齐：>0.85 确认 / >=0.6 候选 / <0.6 丢弃
    step_6  三分支处理：确认则继续 / 有候选则反问 / 无结果则拒识
    step_7  把商品名、改写问题写回历史记录

【重要】本节点是单点入口，返回值会与后续并行分支合并进同一 state，
只返回本节点负责的字段，不要 return state。
"""
import json
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client  # 注意：本项目文件名是 llm_utils（双 L）
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task
from app.utils.tenant_guard import require_tenant_filter

# --- 评分阈值（对齐规则的核心参数）---
# > 0.85：极高置信度，系统认为用户输入与库中名称几乎一致，直接锁定
# >= 0.6：中等置信度，可能是相关商品也可能是误匹配，作为候选让用户确认
# < 0.6 ：低置信度，视为噪音丢弃
HIGH_CONF_THRESHOLD = 0.85
MID_CONF_THRESHOLD = 0.6
# 单个商品名最多返回的匹配结果数
MATCH_TOP_K = 5
# 候选列表最多展示几个（太多反而让用户选择困难）
MAX_OPTIONS_SHOWN = 3
# 选品反问的最多轮次：同一会话连续反问达到该次数仍未确认时，
# 不再纠缠型号，直接走「全库通用检索」，避免用户问系统/概念类问题时被反复拦截
MAX_CLARIFY_ROUNDS = 2
# 反问话术的特征串（用于统计历史里已经反问过几次）
CLARIFY_MARK = "您是想问以下哪个产品"


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState) -> dict:
    """
    主节点函数：商品名称确认流程

    :param state: 会话状态，含 session_id / original_query / is_stream
    :return: 更新后的字段字典（item_names / rewritten_query / answer / history）
    """
    logger.info(">>> node_item_name_confirm: 开始处理")

    session_id = state.get("session_id", "")
    original_query = state.get("original_query", "")

    add_running_task(session_id, sys._getframe().f_code.co_name)

    # 1. 取历史会话（供指代消解用）
    history = step_1_get_history(session_id)

    # 2. 保存用户当前问题（先粗存，step_7 会带着商品名回来更新）
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))

    # 3. LLM 提取商品名 + 改写问题（同时判定 product / general 问题类型）
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
    query_type = (extract_res.get("query_type") or "").strip().lower()

    # 4 & 5. 只有「疑似产品类问题且提取到商品名」才走向量检索与对齐
    #       query_type=general 时直接走通用检索，避免「系统架构是什么」这类问题
    #       被误当成型号、反复反问用户选产品（选品逃生分支）
    align_result: Dict[str, Any] = {}
    if query_type == "general":
        logger.info("Node:问题类型判定为 general，跳过商品名对齐，走通用检索")
    elif item_names:
        query_results = step_4_vectorize_and_query(item_names, state.get("tenant_id", ""))
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node:未提取到商品名，跳过向量检索")

    # 6. 按对齐结果三分支处理
    updates = step_6_check_confirmation(
        align_result, session_id, history, rewritten_query
    )
    # 记录问题类型，供下游（联网按需触发等）参考
    if query_type in ("product", "general"):
        updates["query_type"] = query_type

    # 7. 把改写问题与商品名写回历史
    step_7_write_history(
        session_id, original_query, rewritten_query,
        updates.get("item_names") or [], updates.get("answer"), message_id,
    )

    # 供下游 node_answer_output 使用的上下文
    updates["history"] = history
    updates["rewritten_query"] = rewritten_query

    add_done_task(session_id, sys._getframe().f_code.co_name)
    logger.info(
        f">>> node_item_name_confirm: 处理结束, item_names={updates.get('item_names')}"
    )
    # 注意：answer 非空时，主图的条件路由会跳过后续检索直接输出
    return updates


@step_log("step_1_get_history")
def step_1_get_history(session_id: str) -> List[Dict[str, Any]]:
    """
    步骤1：取最近的历史会话，用于指代消解

    :param session_id: 会话唯一标识
    :return: 历史消息列表；读取失败返回空列表（不应阻断检索）
    """
    try:
        history = get_recent_messages(session_id, limit=10)
        logger.info(f"step_1: 获取到 {len(history)} 条历史消息")
        return history
    except Exception as exc:
        logger.warning(f"step_1: 读取历史失败，按无历史处理：{exc}")
        return []


@step_log("step_3_extract_info")
def step_3_extract_info(query: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    步骤3：LLM 提取商品名并改写问题

    改写的三个理由：
        1. 修复不完美输入：口语化/模糊/错别字/信息缺失（「这个怎么用」缺主语）
        2. 适配检索规则：用户说「咋弄」，知识库写「如何操作」，术语不对齐召回就低
        3. 增强上下文关联：多轮里用户只说「再说说这个」，需结合历史补全

    :param query: 用户原始问题
    :param history: 历史消息列表
    :return: {"item_names": [...], "rewritten_query": "..."}
    """
    # 构造历史对话文本，供 LLM 做上下文理解
    history_text = "".join(
        f"{msg.get('role')}: {msg.get('text')}\n" for msg in (history or [])
    )

    # P1-6：LLM 偶发返回非 JSON（截断/多行噪声）不应一击必杀，允许重试。
    # 最多尝试 MAX_ATTEMPTS 次（首次 + 重试），重试次数用尽才降级返回原始问题。
    client = get_llm_client(json_mode=True)
    # 提示词里的 JSON 花括号用 {{ }} 转义，否则 str.format 会把它当占位符
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text=history_text,
        query=query,
    )
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt),
    ]

    MAX_ATTEMPTS = 3
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.invoke(messages)
            content = response.content or ""
            # 剥离模型可能加的 ```json 代码块围栏
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            content = content.strip()

            result = json.loads(content)
            # 健壮性：字段缺失时补默认值，不让单次异常毁掉整条链路
            result.setdefault("item_names", [])
            result.setdefault("rewritten_query", query)
            # query_type 是新增字段：模型没返回时留空，由后续逻辑按「有商品名」推断
            query_type = str(result.get("query_type") or "").strip().lower()
            result["query_type"] = query_type if query_type in ("product", "general") else ""
            # item_names 必须是字符串列表（模型偶尔会返回嵌套结构）
            if not isinstance(result["item_names"], list):
                result["item_names"] = []
            result["item_names"] = [
                str(n).strip() for n in result["item_names"] if str(n).strip()
            ]
            logger.info(
                f"step_3: 第 {attempt} 次尝试提取成功：商品名 {result['item_names']}，"
                f"改写后问题：{result['rewritten_query']}"
            )
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    f"step_3: 第 {attempt}/{MAX_ATTEMPTS} 次提取失败，将重试：{exc}"
                )
            else:
                logger.error(f"step_3: 重试 {MAX_ATTEMPTS} 次后仍失败：{last_exc}")
                return {"item_names": [], "rewritten_query": query}

    return {"item_names": [], "rewritten_query": query}


@step_log("step_4_vectorize_and_query")
def step_4_vectorize_and_query(item_names: List[str], tenant_id: str) -> List[Dict[str, Any]]:
    """
    步骤4：商品名向量化 + 在 kb_item_names 中混合检索

    原来这一步完全不带过滤条件（expr 缺省），任何调用方问一个模糊商品名就能把
    所有租户的主体名捞出来——既是跨租户信息泄漏，也让 A 的商品名被误认成 B 的。

    :param item_names: LLM 提取的商品名列表
    :param tenant_id: 租户标识，必填位置参数
    :return: [{extracted_name, matches: [{item_name, score}, ...]}, ...]
    """
    results: List[Dict[str, Any]] = []
    if not item_names:
        return results

    tenant_expr = require_tenant_filter(tenant_id)

    client = get_milvus_client()
    if not client:
        logger.error("step_4: 无法连接 Milvus")
        return results

    collection_name = milvus_config.item_name_collection

    try:
        # 批量生成向量比逐个生成效率高
        embeddings = generate_embeddings(item_names)
    except Exception as exc:
        logger.error(f"step_4: 商品名向量化失败：{exc}")
        return results

    for i, name in enumerate(item_names):
        try:
            dense_vec = embeddings["dense"][i]
            sparse_vec = embeddings["sparse"][i]

            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=tenant_expr,
                limit=MATCH_TOP_K,
            )
            # 权重偏向稠密向量(0.8/0.2)：商品型号对齐更依赖语义而非字面
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                limit=MATCH_TOP_K,
                norm_score=True,
                output_fields=["item_name"],
            )

            matches = []
            # Milvus 混合搜索返回二维结构：res[0] 才是本次查询的结果集
            if res and len(res) > 0:
                for hit in res[0]:
                    matched_name = (hit.get("entity", {}) or {}).get("item_name")
                    if not matched_name:
                        continue
                    matches.append(
                        {"item_name": matched_name, "score": hit.get("distance")}
                    )
            results.append({"extracted_name": name, "matches": matches})
            logger.info(f"step_4: '{name}' 匹配到 {len(matches)} 个标准名")
        except Exception as exc:
            # 单个商品名失败不影响其它商品名
            logger.error(f"step_4: 查询商品 '{name}' 出错：{exc}")

    return results


@step_log("step_5_align_item_names")
def step_5_align_item_names(query_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    步骤5：按评分对齐商品名（规则 a>b>c>d，命中即停）

        a. 只有 1 个 >0.85 的结果 → 直接确认
        b. 有多个 >0.85 的结果   → 优先取与原始提取名相同的，否则取最高分
        c. 无 0.85 以上、但有>=0.6 → 取前 5 作为候选
        d. 连 0.6 都没有          → 丢弃（确认与候选均为空）

    :param query_results: step_4 的返回结果
    :return: {"confirmed_item_names": [...], "options": [...]}
    """
    confirmed: List[str] = []
    options: List[str] = []

    for res in query_results or []:
        extracted_name = (res.get("extracted_name") or "").strip()
        matches = res.get("matches") or []
        if not matches:
            continue

        # 按评分降序（高分优先）
        matches.sort(key=lambda x: x.get("score") or 0, reverse=True)
        high = [m for m in matches if (m.get("score") or 0) > HIGH_CONF_THRESHOLD]
        mid = [m for m in matches if (m.get("score") or 0) >= MID_CONF_THRESHOLD]

        # 规则a：唯一的高置信度结果，直接确认
        if len(high) == 1:
            confirmed.append(high[0]["item_name"])
            continue

        # 规则b：多个高置信度，优先选与原始提取名一致的（避免「同系列不同型号」错选）
        if len(high) > 1:
            picked = next(
                (m for m in high if m.get("item_name") == extracted_name), None
            ) or high[0]
            confirmed.append(picked["item_name"])
            continue

        # 规则c：无高置信度，取中置信度前 5 作为候选
        if mid:
            options.extend(m["item_name"] for m in mid[:MATCH_TOP_K])
        # 规则d：无任何达标结果 → 什么都不做，自然落入「丢弃」

    # 去重但保持原有顺序（list(set) 会乱序，这里用 dict 保序去重）
    return {
        "confirmed_item_names": list(dict.fromkeys(confirmed)),
        "options": list(dict.fromkeys(options)),
    }


@step_log("step_6_check_confirmation")
def step_6_check_confirmation(
    align_result: Dict[str, Any],
    session_id: str,
    history: List[Dict[str, Any]],
    rewritten_query: str,
) -> Dict[str, Any]:
    """
    步骤6：按对齐结果三分支处理

        分支A（有确认商品）：写 state.item_names，并回填历史中缺商品名的消息
        分支B（仅候选）    ：生成反问句写入 answer，由主图路由直接输出
        分支C（无结果）    ：生成拒识句写入 answer

    回填历史的用意：用户在确认商品之前的历史提问（含本轮）当时没能确定商品，
    现在确定了，把它们都关联上，后续追问时上下文能保持一致。

    :return: 需要合并进 state 的字段字典
    """
    confirmed = align_result.get("confirmed_item_names") or []
    options = align_result.get("options") or []

    # ---- 分支A：已确认 ----
    if confirmed:
        ids_to_update = [
            str(msg.get("_id"))
            for msg in (history or [])
            if not msg.get("item_names") and msg.get("_id")
        ]
        if ids_to_update:
            try:
                updated = update_message_item_names(ids_to_update, confirmed)
                logger.info(f"step_6: 回填了 {updated} 条历史消息的商品名")
            except Exception as exc:
                logger.warning(f"step_6: 回填历史商品名失败（不影响主流程）：{exc}")
        return {
            "item_names": confirmed,
            "rewritten_query": rewritten_query,
            "has_product": True,
        }

    # ---- 分支B：有候选但不确定，反问用户 ----
    if options:
        # 逃生分支：已经反问过足够多轮仍无法确定型号，说明用户可能问的压根不是产品，
        # 此时不再反问（否则会把「问系统本身」的用户困在选择题里），直接全库检索
        asked_rounds = sum(
            1
            for msg in (history or [])
            if msg.get("role") == "assistant" and CLARIFY_MARK in (msg.get("text") or "")
        )
        if asked_rounds >= MAX_CLARIFY_ROUNDS:
            logger.info(
                f"step_6: 已反问 {asked_rounds} 轮仍未确认商品，触发选品逃生分支，"
                f"改为通用检索"
            )
            return {
                "item_names": [],
                "rewritten_query": rewritten_query,
                "has_product": False,
            }

        shown = "、".join(options[:MAX_OPTIONS_SHOWN])
        return {
            "answer": f"{CLARIFY_MARK}：{shown}？请明确一下型号。",
            "item_names": [],
            "rewritten_query": rewritten_query,
            "has_product": False,
        }

    # ---- 分支C：查无此人 ----
    # 不再拦截，标记 has_product=False 让后续检索节点走通用模式
    logger.info("step_6: 未匹配到商品，走通用检索模式")
    return {
        "item_names": [],
        "rewritten_query": rewritten_query,
        "has_product": False,
    }


@step_log("step_7_write_history")
def step_7_write_history(
    session_id: str,
    original_query: str,
    rewritten_query: str,
    item_names: List[str],
    answer: str,
    message_id: str,
) -> None:
    """
    步骤7：把本轮结果写回历史

    只做一件事：用改写问题 + 最终确认的商品名**更新用户消息**
    （覆盖 step_2 的粗存），让历史里这条提问带上正确的商品归属。

    【不做的事】反问/拒识语（answer）由node_answer_output 统一存档——
    它是答案的唯一出口，无论答案来自本节点的分支 B/C 还是最终生成。
    若这里也存一份，同一次反问会在 MongoDB 里落两条相同 assistant 记录。

    :param message_id: step_2 保存用户消息时返回的主键，用于定位更新
    """
    # answer 非空时不在此存档，交由 node_answer_output 统一处理
    try:
        save_chat_message(
            session_id=session_id,
            role="user",
            text=original_query,
            rewritten_query=rewritten_query,
            item_names=item_names,
            message_id=message_id,
        )
    except Exception as exc:
        logger.warning(f"step_7: 更新用户消息失败（不影响主流程）：{exc}")


# ===================== 本地测试 =====================
if __name__ == "__main__":
    import json as _json

    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False,
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        result_state = node_item_name_confirm(dict(mock_state))
        print("\n>>> 测试完成！最终状态:")
        print(_json.dumps(result_state, indent=2, ensure_ascii=False, default=str))

        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名，answer={result_state.get('answer')}")
    except Exception as exc:
        print(f"\n[FAIL] 测试运行出错: {exc}")