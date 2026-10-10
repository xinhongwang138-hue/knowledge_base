"""
Milvus 字符串安全转义工具

背景：Milvus 的 filter 表达式是字符串解析的（形如 item_name == 'xxx'），
      而 item_name 来自大模型输出或文件名，可能含单引号、双引号、反斜杠等特殊字符。
      若直接拼进表达式，轻则查询不到数据，重则表达式解析报错。

策略：把「单引号字符串」内的内容转义成 Milvus 可解析的安全形式，
      同时处理反斜杠，保证 filter 表达式始终可被正确解析。
"""


def escape_milvus_string(value) -> str:
    """
    转义 Milvus filter 表达式中的字符串值

    转义顺序有讲究，必须先处理反斜杠再处理引号，否则会重复转义：
        1. 反斜杠 \\ -> \\\\\   （转义自身）
        2. 单引号 '  -> \\'     （字符串定界符，最危险的一种）
        3. 双引号 "  -> \\"
        4. 换行/制表等控制字符统一转成空格，避免破坏表达式结构

    :param value: 待转义的原始字符串（None 转为空串）
    :return: 转义后的安全字符串，可直接嵌入单引号字符串字面量中
    """
    if value is None:
        return ""

    text = str(value)
    # 1. 反斜杠最先处理（它本身是转义符，必须先转，否则会影响后续引号的转义）
    text = text.replace("\\", "\\\\")
    # 2. 单引号是 Milvus 字符串的定界符，必须转义，否则会提前闭合表达式
    text = text.replace("'", "\\'")
    # 3. 双引号一并转义，兼容表达式中双引号写法
    text = text.replace('"', '\\"')
    # 4. 控制字符换成空格，防止换行等破坏表达式结构
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    return text


if __name__ == "__main__":
    demos = [
        "华为Mate60 Pro手机",
        "苏泊尔电磁炉（5000W）",
        " O'Reilly 品牌",
        '带"双引号"的名称',
        "反斜杠\\测试",
        "多行\n文本",
        None,
    ]
    for d in demos:
        safe = escape_milvus_string(d)
        expr = f"item_name == '{safe}'"
        print(f"原始: {d!r}")
        print(f"转义: {safe!r}")
        print(f"表达式: {expr}\n")