"""把 Markdown 里夹带的 HTML 表格摊平成 Markdown 表格

问题背景：
    MinerU 把 PDF 里的表格识别成 `<table><tr><td>…` 的原始 HTML 塞进 Markdown。
    这种片段既没有换行也没有中文标点，是一整块几千字符的"不可切分文本"：

        |<tr><td>框架L单元</td><td>×</td><td>○</td>…（实测 1386 字、0 换行、0 句号）

    后果有两个：
        1. 切分器按「段落 → 换行 → 句号」都切不动，`CHUNK_SIZE` 对表格完全失效，
           一整张表压成一个超长切片，向量被稀释，问表里某一格的问题检索不到；
        2. 表格里的关键词（如标准号 SJ/T 11363）被埋在 HTML 标签中间，
           关键词召回基本失效。

    摊平成 Markdown 表格后，每行一个 `| 单元格 | 单元格 |`，既有换行可切，
    单元格内容也能被正常匹配到。

处理策略：
    · 只处理 `<table>` 开闭标签完整的块，标签不完整的原样保留（宁可不动，不能切坏）；
    · 取第一行作表头，`colspan` 只保留文本一次（不补齐空列，避免造出假数据）；
    · 单元格内的行内标签先剥掉，HTML 实体反转义，`|` 转义成 `\\|`。
"""
import html
import re
from typing import List

from app.core.logger import logger

_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table\s*>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RE = re.compile(r"\s+")


def _clean_cell(raw: str) -> str:
    """剥掉单元格里的行内标签、反转义实体、压掉多余空白"""
    text = _TAG_RE.sub(" ", raw or "")
    text = html.unescape(text)
    # 竖线会把 Markdown 表格切断，必须转义
    return _BLANK_RE.sub(" ", text).strip().replace("|", "\\|")


def _table_to_markdown(table_html: str) -> str:
    """
    单个 <table> 转 Markdown

    :param table_html: 表格内部 HTML（不含 <table> 标签本身）
    :return: Markdown 表格文本；解析不出任何单元格时返回空串
    """
    rows: List[List[str]] = []
    for row_html in _ROW_RE.findall(table_html):
        cells = [_clean_cell(c) for c in _CELL_RE.findall(row_html)]
        # 整行都是空的（colspan 占位行）没有信息量，跳过
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    # 每行补齐到同样列数，否则 Markdown 表格渲染会错位
    rows = [row + [""] * (width - len(row)) for row in rows]

    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def html_tables_to_markdown(md_content: str) -> str:
    """
    把文本里所有完整的 HTML 表格换成 Markdown 表格

    :param md_content: 原始 Markdown
    :return: 替换后的 Markdown；没有表格或表格不完整时原样返回
    """
    if not md_content or "<table" not in md_content.lower():
        return md_content

    def _replace(match: re.Match) -> str:
        markdown = _table_to_markdown(match.group(1))
        # 解析不出来就保留原 HTML：切不动总比丢内容好
        return markdown if markdown else match.group(0)

    converted = _TABLE_RE.sub(_replace, md_content)
    if converted != md_content:
        tables = len(_TABLE_RE.findall(md_content))
        logger.info(f"已把 {tables} 个 HTML 表格摊平成 Markdown 表格")
    return converted


if __name__ == "__main__":
    demo = (
        '<p>下表列出限用物质</p><table border="1"><tr><td colspan="6">有害物质</td></tr>'
        "<tr><td>部件名称</td><td>铅(Pb)</td><td>汞(Hg)</td></tr>"
        "<tr><td>框架L单元</td><td>×</td><td>○</td></tr></table>"
    )
    out = html_tables_to_markdown(demo)
    print(out)
    print("\n可切分行数:", len(out.split("\n")))
    print("未闭合表格保持原样:", html_tables_to_markdown("<table><tr><td>x</td>"))
