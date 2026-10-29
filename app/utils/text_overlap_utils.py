# -*- coding: utf-8 -*-
"""切片拼接时去掉重叠段

切片之间带 `CHUNK_OVERLAP` 的重叠，直接拼接会把切口处的文字拼两遍：
不只是难看——重复片段会让向量偏向切口附近，答案里也会出现重复句子。

生产切分（`node_document_split` 把被表格切点分开的片段拼回去）和存量重切
（`scripts/reindex_chunks.py` 从库里读回碎片重组小节）都需要这件事，
所以实现只留这一份，避免两处各写一套慢慢漂移。
"""
from typing import Optional


def strip_overlap(acc: str, nxt: str, max_overlap: Optional[int] = None) -> str:
    """
    去掉 `nxt` 开头与 `acc` 尾部重叠的部分

    做法：找「acc 的后缀 == nxt 的前缀」的最长 k，把 nxt 的前 k 个字去掉。
    必须从长到短找：重叠段本身往往以 `|`、换行这类高频字符结尾/开头，
    从短往长找会先匹配上一个字符就收手，反而把正文吃掉一个字符。

    :param acc: 已累积的正文
    :param nxt: 下一片正文
    :param max_overlap: 重叠上限，默认不限制（只受两段自身长度约束）
    :return: 去掉重叠后的下一片
    """
    if not acc or not nxt:
        return nxt
    limit = len(nxt) if max_overlap is None else max_overlap
    max_k = min(len(acc), len(nxt), limit)
    for k in range(max_k, 0, -1):
        if acc.endswith(nxt[:k]):
            return nxt[k:]
    return nxt
