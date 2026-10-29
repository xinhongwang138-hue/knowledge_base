"""主体名（item_name）规范化工具

问题背景：
    item_name 由大模型自由生成，同一款产品在不同文档里会被起不同的名字。
    实测一个租户里同一台机器有 4 个名字：

        HAK180烫金机 / Brother打印机HAK180 / hak180产品安全手册 / 熔带更换工作台

    查询侧靠「提取名 → 在主体名库里找最相近的标准名」来确认用户问的是哪个产品。
    名字一散，提取名和每个名字的相似度都落在中间区间（0.6~0.85），
    既够不上「确认」，又都够得上「候选」，于是系统判定为歧义，直接反问用户。
    用户明明说得清清楚楚，却被要求「请明确一下型号」——这是这套规则最难受的失效方式。

解法（确定性规则，不用再调一次大模型）：
    型号 token 才是产品的稳定标识。"HAK180烫金机" 和 "Brother打印机HAK180"
    的措辞完全不同，但都含 HAK180。因此：

        1. 从名字里抽型号 token（字母+数字的组合，如 HAK180、ZX900）；
        2. 同一租户下 token 有交集的多个名字，视为同一产品的不同叫法；
        3. 组内选一个「规范名」——取最短的（型号+品类通常最短，噪声最少）。

    没有型号 token 的名字（如「熔带更换工作台」）不参与归并：它根本不是产品名，
    归并进来只会把不相关的文档串到一起。
"""
import re
from typing import Dict, Iterable, List

# 型号 token：字母打头、带数字，允许中间有连字符或空格（HAK180 / ZX-900 / RS 12）
# 只认「字母+数字」的组合：纯数字（180）或纯字母（Brother）单独出现都不足以定位型号
_MODEL_TOKEN_RE = re.compile(r"[A-Za-z]+[\s\-]?\d+[A-Za-z0-9]*")


def extract_model_tokens(name: str) -> List[str]:
    """
    抽取名字里的型号 token（大写、去分隔符）

    :param name: 主体名，如 "Brother打印机HAK180"
    :return: token 列表，如 ["HAK180"]；没有型号 token 时返回空列表
    """
    if not name:
        return []
    tokens: List[str] = []
    for raw in _MODEL_TOKEN_RE.findall(name):
        # 统一大小写、去掉连字符和空格，保证 "ZX-900" 与 "zx900" 视为同一个
        token = re.sub(r"[\s\-]", "", raw).upper()
        if token not in tokens:
            tokens.append(token)
    return tokens


def group_by_model(names: Iterable[str]) -> Dict[str, List[str]]:
    """
    按型号 token 把名字分组

    一个名字可能同时含多个 token，这种情况下它会被放进每个对应分组；
    调用方按「token → 名字列表」使用即可，不需要保证互斥。

    :param names: 名字集合
    :return: {token: [名字, ...]}；没有 token 的名字不出现在结果里
    """
    groups: Dict[str, List[str]] = {}
    for name in names:
        for token in extract_model_tokens(name):
            groups.setdefault(token, []).append(name)
    return groups


def _is_bare_model(name: str) -> bool:
    """
    判断名字是不是「只有型号、没有品类」

    "HAK180" 这种纯型号不能当规范名：它比 "HAK180烫金机" 短，会赢下「最短优先」的
    挑选规则，结果所有同型号文档都被贴上光秃秃的型号，用户看不出那是什么东西。

    :param name: 主体名
    :return: 整个名字就是一个型号 token 时返回 True
    """
    tokens = extract_model_tokens(name)
    if len(tokens) != 1:
        return False
    return re.sub(r"[\s\-]", "", name).upper() == tokens[0]


def pick_canonical(names: Iterable[str]) -> str:
    """
    从同一产品的多个叫法里选一个规范名

    规则：先排除「只有型号」的名字，再最短优先（型号+品类通常最短、噪声最少），
    长度相同按字典序，保证结果稳定。

    :param names: 同一产品的名字集合
    :return: 规范名；入参全为空时返回空串
    """
    candidates = [n.strip() for n in names if n and n.strip()]
    if not candidates:
        return ""
    with_genre = [n for n in candidates if not _is_bare_model(n)]
    return sorted(with_genre or candidates, key=lambda n: (len(n), n))[0]


def canonicalize(name: str, existing_names: Iterable[str]) -> str:
    """
    把新识别的名字对齐到已有规范名

    仅当两边存在共同型号 token 时才归并 —— 这是「同一产品」的确定性证据。
    没有共同 token（或新名字本身没有 token）时原样返回，绝不猜。

    :param name: 新识别出的主体名
    :param existing_names: 同租户下已有的主体名
    :return: 对齐后的名字（可能是 existing 里的规范名，也可能是原名）
    """
    name = (name or "").strip()
    if not name:
        return ""

    new_tokens = set(extract_model_tokens(name))
    if not new_tokens:
        # 连型号都没有的名字（"熔带更换工作台"），没有归并依据
        return name

    same_product = [
        other for other in existing_names
        if other and other.strip() and new_tokens & set(extract_model_tokens(other))
    ]
    if not same_product:
        return name

    # 候选集合包含新名字本身：若新名字比历史名更短更规范，就采用新名字
    return pick_canonical(same_product + [name])


if __name__ == "__main__":
    # 用真实数据自测：这 4 个名字来自同一个租户，应当收敛到 HAK180烫金机
    existing = [
        "HAK180烫金机",
        "Brother打印机HAK180",
        "hak180产品安全手册",
        "熔带更换工作台",
        "ZX-900点胶机",
    ]
    print("=== 型号 token ===")
    for name in existing:
        print(f"  {name:24s} -> {extract_model_tokens(name)}")

    print("\n=== 归并结果 ===")
    for name in ["HAK-180 烫金机", "Brother打印机HAK180", "导热辊清洁", "ZX900 点胶机"]:
        print(f"  {name:20s} -> {canonicalize(name, existing)}")

    print("\n=== 分组 ===")
    print(" ", group_by_model(existing))
