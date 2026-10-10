import uuid
from pathlib import Path

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> dict:
    """
    节点：入口节点(node_entry)
    职责：
      1. 生成 task_id（后续节点追踪用）
      2. 校验输入文件路径
      3. 判定走 PDF 导入还是 MD 直接导入路径
      4. 提取文件标题
    按 LangGraph 规范，只返回「需要更新的字段」，不改动传入的 state。
    """
    local_file_path = state.get("local_file_path", "")
    # task_id 由入口节点生成；已有则沿用（如断点续跑）
    task_id = state.get("task_id") or uuid.uuid4().hex

    if not local_file_path:
        logger.warning("没有输入文件地址，无法处理，流程终止")
        return {"task_id": task_id}

    # 有实际输入才算「任务开始」
    add_running_task(task_id, "node_entry")

    suffix = Path(local_file_path).suffix.lower()

    # MD 直接导入：跳过 PDF转MD
    if suffix == ".md":
        logger.info(f"识别为 Markdown 文件，走 MD 直接导入路径：{local_file_path}")
        add_done_task(task_id, "node_entry")
        return {
            "task_id": task_id,
            "is_md_read_enabled": True,
            "md_path": local_file_path,
            "file_title": Path(local_file_path).stem,
        }

    # PDF 导入：先转 Markdown
    if suffix == ".pdf":
        logger.info(f"识别为 PDF 文件，走 PDF 导入路径：{local_file_path}")
        add_done_task(task_id, "node_entry")
        return {
            "task_id": task_id,
            "is_pdf_read_enabled": True,
            "pdf_path": local_file_path,
            "file_title": Path(local_file_path).stem,
        }

    # 不支持的类型：只告警并返回，不再往下走。
    # 注意这里不能用 `elif logger.warning(...)`——日志函数的返回值恒为 None，
    # 那个条件永远为 False，会导致本该终止的分支被跳过。
    logger.warning(f"无法识别文件类型，目前只支持 md 和 pdf：{local_file_path}")
    add_done_task(task_id, "node_entry", status="failed")
    return {"task_id": task_id}





if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")