import copy
from typing import TypedDict

from app.core.logger import logger


class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["local_file_path"], state.get("chunks")）
    """
    task_id: str    # 任务唯一ID，用于追踪日志
    tenant_id: str  # 租户标识，写入 Milvus 作 partition key，检索时强制过滤

    # ---流程控制标记---
    is_md_read_enabled: bool    # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool   # 是否启用 PDF 读取路径

    # ---路径相关---
    local_dir: str      # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str     # 文件标题(文件名去后缀)
    pdf_path: str       # PDF 文件路径(如果输入是 PDF)
    md_path: str        # Markdown 文件路径(转换后或直接输入的)

    # ---内容数据---
    md_content: str     # Markdown 的全文内容
    chunks: list        # 切片后的文本列表，包含 metadata
    item_name: str      # 识别出的主体名称(如："万用表")，用于增强检索

    # ---数据库相关---
    embeddings_content: str  # 包含向量数据的列表，准备写入 Milvus


# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id": "",
    "tenant_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持用关键字参数覆盖任意字段
    Args:
        **overrides: 要覆盖的字段(关键字参数解包)
    Returns:
        新的状态实例(深拷贝，避免污染全局默认值)
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == '__main__':
    # 测试
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)