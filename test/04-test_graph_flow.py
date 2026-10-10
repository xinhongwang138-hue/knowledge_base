import json
import sys
from pathlib import Path

# sys.path 引导：把项目根目录插到最前面，保证脱离 PyCharm 也能 import app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state

logger.info("==开始测试==")

initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")

executed_nodes = []
final_state = None

# 同时订阅两种流模式，单次遍历即可拿到「节点名」和「全量状态快照」：
#   updates -> {节点名: 该节点的增量更新}，节点返回 {} 时值为 None
#   values  -> 当前完整状态字典，最后一个即为最终状态
for mode, chunk in kb_import_app.stream(
    initial_state, stream_mode=["updates", "values"]
):
    if mode == "updates":
        for node_name, delta in chunk.items():
            executed_nodes.append(node_name)
            logger.info(f"节点：{node_name} -> 状态更新：{delta}")
    else:
        final_state = chunk

logger.info(f"实际执行节点（共 {len(executed_nodes)} 个）：{executed_nodes}")
logger.info(f"最终状态：{json.dumps(final_state, indent=4, ensure_ascii=False, default=str)}")

logger.info("图结构")
kb_import_app.get_graph().print_ascii()

logger.info("=测试结束=")