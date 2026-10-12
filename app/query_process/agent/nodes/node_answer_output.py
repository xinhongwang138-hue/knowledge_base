"""
节点：答案生成（node_answer_output）

职责：检索链路的最后一公里 —— 把前面各节点的成果组装成 Prompt，
交给大模型生成最终答案，并通过 SSE 推给前端。

五个动作：
    1. 检查前置答案：主体确认节点可能已生成反问/拒识语（分支B/C），
       有则直接输出，跳过 LLM 生成 —— 这既省一次模型调用，也让响应更快
    2. 组装 Prompt：问题 + 历史对话 + 商品名 + Rerank 后的 TopK 文档
    3. LLM 生成：流式模式逐字推送 delta（打字机效果），非流式一次性返回
    4. 图片提取：从文档正文里正则抽出图片 URL 补进答案
    5. 收尾：写MongoDB + 推送 FINAL 事件（触发前端渲染图片）

【重要】本节点返回增量字典，与其它节点的写法保持一致。
"""
import re
import sys
from typing import List

from app.clients.minio_utils import presign_public_image_url
from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import add_done_task, add_running_task, set_task_result

# 组装 Prompt 时参考内容的字符上限，防止撑爆 LLM 的上下文窗口
MAX_CONTEXT_CHARS = 12000
# Markdown 图片语法：![alt](url)，非贪婪捕获 url 部分
MD_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
# 常见图片后缀（用于校验 url 字段是否直接指向图片）
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> dict:
    """
    答案生成节点

    :param state: 含 session_id / original_query / rewritten_query / history /
                  item_names / reranked_docs / answer / is_stream
    :return: {"answer": "...", "prompt": "..."}
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    session_id = state.get("session_id", "")
    add_running_task(session_id, sys._getframe().f_code.co_name)

    answer_exists = step_1_check_answer(state)

    if not answer_exists:
        # 阶段二：组装 Prompt
        prompt = step_2_construct_prompt(state)
        # 阶段三：调用大模型生成
        step_3_generate_response(state, prompt)

    # 图片提取（无论答案来自前置分支还是 LLM，都需要提取）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
    # P1-5 预签名改造：私有桶下历史/存量直链无法匿名访问，
    # 对外返回前实时重签为预签名 URL（保持字段名与结构不变）
    image_urls = _presign_image_url_list(image_urls)

    # 阶段四：写入历史（answer 文本内嵌的图片 URL 也重签后再落库）
    if state.get("answer"):
        state["answer"] = _presign_text_image_urls(state.get("answer", ""))
        step_4_write_history(state, image_urls)

    add_done_task(session_id, sys._getframe().f_code.co_name)

    # 阶段五：推送 FINAL 事件，让前端渲染图片
    if state.get("is_stream"):
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": state.get("answer", ""),
                "status": "completed",
                "image_urls": image_urls,
            },
        )

    logger.info("---node_answer_output 节点处理结束---")
    return {"answer": state.get("answer", ""), "prompt": state.get("prompt", "")}


@step_log("step_1_check_answer")
def step_1_check_answer(state: QueryGraphState) -> bool:
    """
    阶段一：检查 state 中是否已有答案

    主体确认节点在「多选一反问」与「查无此人拒识」两种场景下会直接写 answer，
    此时无需再调 LLM 生成。

    :param state: 图状态（会被就地更新 is_stream 分支下的 task_result）
    :return: True 表示已有答案（已按需推送），False 表示需要走 LLM 生成
    """
    answer = state.get("answer")
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    if not answer:
        return False

    if is_stream:
        logger.info("step_1: 已有前置答案，直接流式推送")
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
    else:
        # 非流式模式让 query_service 能同步取到答案
        set_task_result(session_id, answer=answer)
    return True


@step_log("step_2_construct_prompt")
def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    阶段二：组装 Prompt

    参考内容的拼装策略：
        1. 按 Rerank 后的相关性顺序逐条取文档
        2. 每条加上元数据标签（序号/来源/chunk_id/分数/标题），让 LLM 知道出处
        3. 累加字符数，超 MAX_CONTEXT_CHARS 即停止，防止撑爆上下文窗口

    :param state: 图状态
    :return: 组装好的 prompt 文本
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""
    history = state.get("history") or []
    item_names = state.get("item_names") or []
    reranked_docs = state.get("reranked_docs") or []

    # ---- 参考内容 ----
    docs: List[str] = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        meta = [f"[{i}]"]
        if doc.get("source"):
            meta.append(f"[{doc['source']}]")
        if doc.get("chunk_id"):
            meta.append(f"[chunk_id={doc['chunk_id']}]")
        if doc.get("url"):
            meta.append(f"[url={doc['url']}]")
        if doc.get("score") is not None:
            meta.append(f"[score={float(doc['score']):.4f}]")
        if doc.get("title"):
            meta.append(f"[title={doc['title']}]")

        block = " ".join(meta) + "\n" + text
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        docs.append(block)
        used += len(block) + 2

    context_str = "\n\n".join(docs) if docs else "无参考内容"

    # ---- 历史对话 ----
    lines = []
    for msg in history:
        role = msg.get("role")
        text = msg.get("text")
        if not text:
            continue
        lines.append(f"{'用户' if role == 'user' else '助手'}: {text}")
        if used + sum(len(x) for x in lines) > MAX_CONTEXT_CHARS:
            break
    history_str = "\n".join(lines) if lines else "无历史对话"

    item_names_str = "、".join(item_names) if item_names else "无指定商品"

    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )
    logger.info(f"step_2: Prompt 组装完成，长度 {len(prompt)}")
    return prompt


@step_log("step_3_generate_response")
def step_3_generate_response(state: QueryGraphState, prompt: str):
    """
    阶段三：调用大模型生成答案（支持流式）

    :param state: 图状态，答案写入 state['answer']
    :param prompt: 阶段二组装的提示词
    """
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")
    llm = get_llm_client()

    if is_stream:
        logger.info(f"step_3: 流式生成中 (session={session_id})")
        final_text = ""
        try:
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    final_text += delta
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
            logger.info(f"step_3: 流式完成，总长度 {len(final_text)}")
        except Exception as exc:
            logger.error(f"step_3: 流式生成出错：{exc}")
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(exc)})
        state["answer"] = final_text
    else:
        logger.info("step_3: 非流式生成中")
        try:
            response = llm.invoke(prompt)
            state["answer"] = response.content or ""
            set_task_result(session_id, answer=state["answer"])
            logger.info(f"step_3: 生成完成，长度 {len(state['answer'])}")
        except Exception as exc:
            logger.error(f"step_3: 生成出错：{exc}")
            state["answer"] = "抱歉，生成回答时出现错误，请稍后重试。"


def _extract_images_from_docs(docs) -> List[str]:
    """
    从文档列表中提取图片 URL

    两个来源：
        1. url 字段直接指向图片（常见于联网搜索结果）—— 校验后缀
        2. text 正文里的 Markdown 图片语法（常见于本地切片）

    为什么不解析 LLM 生成的【图片】区块，而是直接扫源文档：
        LLM 可能编造 URL 或格式写错，而源文档里的图片肯定真实存在。
        Prompt 里的【图片】要求只是引导模型在文字中提及图片（如「请看下图」）。

    :param docs: 文档列表
    :return: 去重后的图片 URL 列表
    """
    images: List[str] = []
    seen = set()

    for i, doc in enumerate(docs or []):
        if not isinstance(doc, dict):
            continue

        # 来源1：url 字段直接是图片
        url = (doc.get("url") or "").strip()
        if url and url.lower().endswith(IMAGE_SUFFIXES) and url not in seen:
            seen.add(url)
            images.append(url)

        # 来源2：正文里的 Markdown 图片
        text = doc.get("text") or ""
        for img_url in MD_IMAGE_PATTERN.findall(text):
            img_url = img_url.strip()
            if img_url and img_url not in seen:
                seen.add(img_url)
                images.append(img_url)

    logger.info(f"图片提取完成，共 {len(images)} 张")
    return images


def _presign_image_url_list(image_urls: List[str]) -> List[str]:
    """
    将图片 URL 列表实时重签为预签名 URL（P1-5 私有桶改造）。

    非本桶 URL 原样保留（presign_public_image_url 内部已做 host/bucket 白名单）；
    重签失败时回退原 URL，避免对外返回空串导致前端图片裂图。

    :param image_urls: 原始图片 URL 列表
    :return: 重签后的 URL 列表（结构/顺序不变）
    """
    presigned = []
    for url in image_urls or []:
        new_url = presign_public_image_url(url)
        presigned.append(new_url or url)  # 重签失败回退原 URL
    if presigned != (image_urls or []):
        logger.info(f"图片 URL 预签名重签完成（{len(presigned)} 条）")
    return presigned


def _presign_text_image_urls(text: str) -> str:
    """
    将答案文本中 Markdown 图片语法内的 URL 实时重签为预签名 URL。

    LLM 生成的答案可能原样引用文档中的图片直链，私有桶下需一并重签，
    保证前端渲染答案内嵌图片时不被 403 阻断。非本桶 URL 不受影响。

    :param text: 答案文本
    :return: 图片 URL 重签后的文本
    """
    if not text:
        return text

    def _replace(match):
        url = match.group(1).strip()
        new_url = presign_public_image_url(url)
        if new_url == url:
            return match.group(0)  # 非本桶/未变化，原样返回
        whole = match.group(0)
        alt = whole[2:whole.index("](")]  # 提取 ![... 中的 alt
        return f"![{alt}]({new_url})"

    return MD_IMAGE_PATTERN.sub(_replace, text)


@step_log("step_4_write_history")
def step_4_write_history(state: QueryGraphState, image_urls: List[str]):
    """
    阶段四：把答案写入 MongoDB

    写历史失败不该影响主链路，故单独 try/except。

    :param state: 图状态
    :param image_urls: 提取到的图片 URL
    """
    answer = (state.get("answer") or "").strip()
    if not answer:
        return
    try:
        save_chat_message(
            session_id=state.get("session_id", "default"),
            role="assistant",
            text=answer,
            rewritten_query="",
            item_names=state.get("item_names") or [],
            image_urls=image_urls,
        )
    except Exception as exc:
        logger.error(f"step_4: 写入历史失败（不影响主流程）：{exc}")


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK180烫金机"],
        "history": [
            {"role": "user", "text": "你好"},
            {"role": "assistant", "text": "请问您需要什么帮助？"},
        ],
        "reranked_docs": [
            {
                "chunk_id": "local_101",
                "source": "local",
                "title": "HAK 180 操作手册",
                "score": 0.95,
                "text": "HAK 180 烫金机的操作面板位于机器正前方。\n"
                        "![操作面板布局图](http://127.0.0.1:9000/kb/images/panel.jpg)\n"
                        "开启电源后先设置温度，建议 110℃ 左右。",
            },
            {
                "chunk_id": None,
                "source": "web",
                "title": "烫金机使用教程",
                "url": "http://example.com/guide",
                "score": 0.72,
                "text": "局部烫金可通过侧面旋钮调节区域。",
            },
        ],
        "is_stream": False,
    }

    try:
        result = node_answer_output(dict(mock_state))
        prompt = result.get("prompt", "")
        answer = result.get("answer", "")
        print(f"\n[{'PASS' if prompt else 'FAIL'}] Prompt 构建（长度 {len(prompt)}）")
        print(f"[{'PASS' if answer else 'FAIL'}] 答案生成（长度 {len(answer)}）")
        print(f"\n答案预览:\n{answer[:200]}")
    except Exception as exc:
        logger.exception(f"测试运行期间发生未捕获异常: {exc}")