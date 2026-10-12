# knowledge_base 冗余垃圾代码审查清单

审查时间：2026-10-12
审查方式：AST 静态分析 + 全项目引用检索 + git 跟踪状态确认（只审查，未删除任何文件）

---

## A 级：可安全删除（确凿冗余，无引用）

| # | 路径 | 冗余类型 | 建议 |
|---|---|---|---|
| A1 | app\query_process\page\chat_20261010_200401_546.html（16KB） | 备份/重复 html | 删除（回收站） |
| A2 | app\query_process\page\chat_backup_20261010_204000.html（16KB） | 备份/重复 html | 删除（回收站） |
| A3 | app\utils\normalize_sparse_vector.py L35-44 `batch_normalize_sparse_vectors` | 死函数（全项目 0 引用） | 删除该函数，保留文件其余部分 |
| A4 | app\utils\task_utils.py L128 `get_task_status` | 死函数（全项目 0 引用） | 删除该函数，保留文件其余部分 |
| A5 | query_service_start.log（0B）+ query_service_start_err.log（77B） | 调试残留空日志 | 删除 |
| A6 | app\conf\（仅 __init__.py 空文件） | 空包目录，0 引用 | 删除目录 |
| A7 | app\tool\（仅 __init__.py 空文件） | 空包目录，0 引用 | 删除目录 |
| A8 | temp\ 下调试残留 | 中间产物 | 删除 |
| A8-1 | temp\query_body_1012.json / query_body_103233.json | 调试请求残留 | 删除 |
| A8-2 | temp\smoke_presign_test.md / smoke_test_101434.md | 冒烟测试输出 | 删除 |
| A8-3 | temp\start_8010.py | 临时启动脚本 | 删除 |
| A8-4 | temp\svc_8010.err.log / svc_8010.out.log / svc_8011.err.log / svc_8011.out.log / svc_8011_restart_102119.*.log | 临时服务日志 | 删除 |
| A8-5 | temp\backup_20261011_222420_*.py（9个）、backup_20261011_230057_*（6个）、backup_20261011_231137_node_web_search_mcp.py、backup_20261011_232326_node_search_embedding.py | 旧版代码备份（git 未跟踪，改造已入库） | 建议删除（如需留底可先归档） |
| A8-6 | temp\backup_20261011_213903 / backup_20261011_214615 / backup_20261011_214848 | 旧备份目录（含旧文件 4 个） | 建议删除（如需留底可先归档） |
| A8-7 | temp\backup_20261012_093038（含 .env 及 5 个改造文件） | 最近一次 P1-5 改造备份 | 建议保留至改造稳定后删除 |

## B 级：需人工确认（有业务含义或正被 git 跟踪）

| # | 路径 | 冗余类型 | 建议 |
|---|---|---|---|
| B1 | output\20261010\（30 文件 175KB，含 测试带图文档/hak180 手册 pdf/zip/json） | 导入测试输出产物 | 确认无业务价值后删除（部分已被 git 跟踪，需 git rm） |
| B2 | output\20261011\（8 文件 273KB，掌柜智库文档导入产物） | 导入测试输出产物 | 确认无业务价值后删除 |
| B3 | output\20261012\（12 文件 10KB，smoke/import_flow/TR-AX3000 等） | 导入测试输出产物 | 确认无业务价值后删除 |
| B4 | output\hak180产品安全手册\（7 文件 5KB）+ output\测试带图文档\（10 文件 130KB） | 测试导入产物（git 已跟踪） | 确认无业务价值后删除 |
| B5 | output\architecture_review_2026-10-11.md | 架构审查报告 | 建议保留（历史文档） |
| B6 | logs\（5 个日志约 2.1MB） | 运行日志（gitignore 已忽略） | 确认服务未运行后删除 |
| B7 | doc\hak180产品安全手册.pdf（602B 占位）+ doc\测试带图文档.pdf（32KB） | 测试输入文档 | 确认 doc 目录用途后决定 |
| B8 | __pycache__\、test\__pycache__\、.idea\（约 16KB） | IDE/缓存（gitignore 已忽略） | 可删可留，不影响仓库 |

## C 级：代码级疑似项（需人工复核，默认保留）

| # | 位置 | 说明 | 建议 |
|---|---|---|---|
| C1 | app\lm\embedding_config.py L55 print(logger_fp16) | 业务路径上的 print（CPU 降级提示） | 改为 logger.info 或保留，不建议直接删 |
| C2 | 多文件 `import app.core.torch_env` / `from app.core.torch_env import TORCH_ENV_INITIALIZED` 提示未使用（main_graph.py、file_import_service.py、query_service.py、embedding_utils.py、reranker_utils.py） | 导入即生效（设置线程环境变量），AST 判为未使用属误报 | 保留，不动 |
| C3 | app\core\logger.py L93 print(..., file=sys.stderr) | 日志初始化失败回退提示 | 保留（合理错误提示） |
| C4 | 各 config/节点文件的 __main__ 自测块（约 40 处） | 本地测试代码，不影响运行 | 保留（开发自测用） |

## 统计

- A 级可安全删除：8 组（含 temp 内 7 小类）
- B 级需确认：8 组
- C 级默认保留：4 组
- 业务代码（app/ 下 50 余个 py、chat.html、import.html、main.py、prompts/ 等）未发现死分支/重复实现，均属正常
