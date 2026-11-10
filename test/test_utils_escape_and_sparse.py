"""纯函数单测：Milvus 字符串转义 与 稀疏向量归一化。

两块都不依赖 Milvus / Mongo / BGE-M3，任何机器上都能离线跑：

    pytest test/test_utils_escape_and_sparse.py -q

为什么先补这两个：它们失败时不抛异常，只让结果悄悄变差。
转义写错会让 filter 表达式提前闭合（查不到数据或解析报错），
归一化写错会让稀疏通道的权重压过稠密通道，融合排序整体失真。
这类「不报错的错」靠人眼看输出发现不了，只能靠断言。
"""

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.normalize_sparse_vector import normalize_sparse_vector


class TestEscapeMilvusString:
    """转义结果必须能安全嵌进 item_name == '...' 的单引号字面量。"""

    def test_none_becomes_empty_string(self):
        assert escape_milvus_string(None) == ""

    def test_plain_text_untouched(self):
        assert escape_milvus_string("华为Mate60 Pro手机") == "华为Mate60 Pro手机"

    def test_single_quote_is_escaped(self):
        assert escape_milvus_string("O'Reilly") == r"O\'Reilly"

    def test_double_quote_is_escaped(self):
        assert escape_milvus_string('带"双引号"') == r'带\"双引号\"'

    def test_backslash_escaped_before_quotes(self):
        # 反斜杠必须先转再用引号规则，否则 \' 会被二次转义成 \\' 以外的形态
        assert escape_milvus_string("a\\b") == "a\\\\b"
        assert escape_milvus_string("a\\'b") == "a\\\\\\'b"

    def test_control_chars_become_spaces(self):
        assert escape_milvus_string("多行\n文本\t带\r回车") == "多行 文本 带 回车"

    def test_non_string_input_is_coerced(self):
        assert escape_milvus_string(123) == "123"

    def test_no_bare_quote_survives(self):
        """核心断言：转义后不存在「前面没有反斜杠」的单引号。

        注入串会试图用 ' 提前闭合表达式，只要还有裸引号，
        item_name == '...' 就会被截断。
        """
        evil = "x' or item_name != '"
        escaped = escape_milvus_string(evil)
        assert not re.search(r"(?<!\\)'", escaped), f"仍有裸单引号: {escaped!r}"

    def test_quote_count_preserved_after_escaping(self):
        # 转义只加反斜杠，不增删字符：引号个数应与原文一致
        evil = "x' or '1'='1"
        assert escape_milvus_string(evil).count("'") == evil.count("'")


class TestNormalizeSparseVector:
    """L2 归一化后模长必须为 1，方向不变，且不修改入参。"""

    def test_empty_dict_returned_as_is(self):
        assert normalize_sparse_vector({}) == {}

    def test_none_returned_as_is(self):
        assert normalize_sparse_vector(None) is None

    def test_unit_norm_after_normalize(self):
        v = {3: 0.7, 8: 0.2, 20: 0.1, 1: 0.6, 9: 0.4}
        n = normalize_sparse_vector(v)
        norm = math.sqrt(sum(w * w for w in n.values()))
        assert math.isclose(norm, 1.0, rel_tol=1e-12)

    def test_direction_preserved(self):
        n = normalize_sparse_vector({1: 3.0, 2: 4.0})
        assert math.isclose(n[1], 0.6, rel_tol=1e-12)
        assert math.isclose(n[2], 0.8, rel_tol=1e-12)

    def test_feature_keys_preserved(self):
        v = {7: 2.0, 9: 2.0, 11: 1.0}
        assert set(normalize_sparse_vector(v)) == {7, 9, 11}

    def test_all_zero_vector_returned_unchanged(self):
        v = {1: 0.0, 2: 0.0}
        assert normalize_sparse_vector(v) == v

    def test_negative_weights_keep_sign(self):
        n = normalize_sparse_vector({1: -3.0, 2: 4.0})
        assert math.isclose(n[1], -0.6, rel_tol=1e-12)
        assert math.isclose(n[2], 0.8, rel_tol=1e-12)

    def test_input_not_mutated(self):
        v = {1: 3.0, 2: 4.0}
        normalize_sparse_vector(v)
        assert v == {1: 3.0, 2: 4.0}
