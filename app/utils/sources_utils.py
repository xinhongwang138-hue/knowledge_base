# -*- coding: utf-8 -*-
"""
引用溯源工具

职责：把 Rerank 后的文档列表压缩成前端可直接渲染的「来源清单」。

设计约定（与 node_answer_output 的 Prompt 拼装严格对齐）：
    · id 必须等于 Prompt 里注入的序号 [i]，前端点 [1] 就能对到来源第 1 条；
      因此这里沿用 enumerate 的原始下标，遇到空正文跳过但不重排序号。
    · 只透出「可展示」字段，不把整段 chunk 正文回传（前端只给摘要预览）。
"""

from typing import Any, Dict, List, Optional
import re

# 默认最多返回的来源条数（与答案节点的上下文条数上限保持一致）
DEFAULT_MAX_SOURCES = 20
# 摘要预览长度（字符）
DEFAULT_SNIPPET_LEN = 200
# Markdown 图片语法：切片正文里内嵌的图片对来源预览没有意义，替换成占位
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Markdown 标题前缀
MD_HEAD_RE = re.compile(r"^#{1,6}\s*")


def _clean_title(title: Any) -> str:
    """切片标题取首行并去掉 Markdown 标题符号，避免来源区显示 '## 设备'"""
    text = str(title or "").strip()
    if not text:
        return ""
    first = text.splitlines()[0]
    return MD_HEAD_RE.sub("", first).strip()[:60]


def _clean_snippet(text: str) -> str:
    """摘要预览清洗：去内嵌图片语法、去行首标题符号、压缩空行与长空白"""
    cleaned = MD_IMAGE_RE.sub("[图片]", text or "")
    cleaned = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", cleaned)
    cleaned = re.sub(r"-{3,}", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _to_float(value: Any) -> Optional[float]:
    """分数转 float，失败则返回 None（避免脏数据打断主链路）"""
    if value is None or value == "":
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def build_sources(
    reranked_docs: Optional[List[Dict[str, Any]]],
    max_sources: int = DEFAULT_MAX_SOURCES,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
) -> List[Dict[str, Any]]:
    """
    把 reranked_docs 转成来源清单

    :param reranked_docs: Rerank 节点产出的文档列表
    :param max_sources: 最多返回条数
    :param snippet_len: 摘要预览字符数
    :return: [{"id","source","title","parent_title","file_title","chunk_id","url","score","snippet"}]
    """
    sources: List[Dict[str, Any]] = []
    for index, doc in enumerate(reranked_docs or [], start=1):
        if not isinstance(doc, dict):
            continue
        text = (doc.get("text") or "").strip()
        if not text:
            # 空正文档不会进入 Prompt，序号自然跳过；后续序号仍按 enumerate 对齐
            continue

        # 先清洗再截断：否则截断会把 ![](url) 切成半截导致正则失配
        cleaned = _clean_snippet(text)
        snippet = cleaned if len(cleaned) <= snippet_len else cleaned[:snippet_len] + "…"
        source_type = (doc.get("source") or "local").strip() or "local"
        title = _clean_title(doc.get("title")) or _clean_title(doc.get("file_title"))
        if not title:
            title = "联网结果" if source_type == "web" else "知识库切片"

        sources.append(
            {
                "id": index,
                "source": source_type,
                "title": title,
                # 章节：title 是切片首个标题，parent_title 是它所属的那一节。
                # 只给 title 的话，出处只能指到"哪个小标题"，指不到"哪一章"。
                "parent_title": _clean_title(doc.get("parent_title")),
                "file_title": (doc.get("file_title") or "").strip(),
                "chunk_id": (doc.get("chunk_id") or ""),
                "url": (doc.get("url") or "").strip(),
                "score": _to_float(doc.get("score")),
                "snippet": snippet,
            }
        )
        if len(sources) >= max_sources:
            break
    return sources
