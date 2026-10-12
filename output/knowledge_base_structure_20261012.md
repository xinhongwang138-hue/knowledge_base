---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 2ef7930562d418b0a12963f90da05d53_27116d61c62611f19b82525400de85a5
    ReservedCode1: n4c36VA/pfPzRzZPiySof39qAj2Pf7uEEP2swAWFXhczt0PomQZtfIIDSCSWwszJ+oJQMeHeefazJKPz7C9Edpp+80fVauvhnwnP9asVejaWKWc8d6ZKDNr/xCyitJGlQADw8Zz4CARxEeQv9qrB3+JsQR0n4KCG4yw/lLHhAaMJ9rEbLZPEXRAQci4=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 2ef7930562d418b0a12963f90da05d53_27116d61c62611f19b82525400de85a5
    ReservedCode2: n4c36VA/pfPzRzZPiySof39qAj2Pf7uEEP2swAWFXhczt0PomQZtfIIDSCSWwszJ+oJQMeHeefazJKPz7C9Edpp+80fVauvhnwnP9asVejaWKWc8d6ZKDNr/xCyitJGlQADw8Zz4CARxEeQv9qrB3+JsQR0n4KCG4yw/lLHhAaMJ9rEbLZPEXRAQci4=
---

# 掌柜智库知识库项目结构大纲

**生成时间**：2026-10-12 16:40  
**项目根目录**：`D:\PythonProject\knowledge_base`

---

## 一、项目概览

| 类别 | 数量 |
|---|---|
| Python 代码文件 (.py) | 43 |
| 前端 HTML 文件 (.html) | 2 |
| 配置文件 (.yml / .env / .toml / .json / .ignore) | 6 |
| Prompt 模板 (.prompt) | 6 |
| 启动脚本 (.ps1 / .bat) | 2 |
| 文档说明 (.md) | 2 |
| 容器文件 (.dockerfile / .dockerignore) | 2 |
| **项目自身文件合计** | **63** |

---

## 二、目录结构

```
knowledge_base/
├── .dockerignore              # Docker 忽略规则
├── .env                       # 环境变量配置（MongoDB/Milvus/MinIO/LLM）
├── .env.bak_before_vl_fix     # .env 备份（VL 修复前）
├── .gitignore                 # Git 忽略规则
├── Dockerfile                 # Docker 镜像构建
├── main.py                    # 统一启动入口（import 服务，端口 8010）
├── pyproject.toml             # Python 项目配置与依赖
├── start_all.bat              # Windows 一键启动脚本（批处理）
├── start_all.ps1              # Windows 一键启动脚本（PowerShell）
├── uv.lock                    # uv 依赖锁定文件
├── README.md                  # 项目说明文档
├── prompts/                   # Prompt 模板（6个）
│   ├── answer_out.prompt      # 答案输出 Prompt
│   ├── hyde_prompt.prompt     # HyDE 检索提示
│   ├── image_summary.prompt   # 图片摘要 Prompt
│   ├── item_name_recognition.prompt
│   ├── product_recognition_system.prompt
│   └── rewritten_query_and_itemnames.prompt
├── test/                      # 测试目录
│   └── 04-test_graph_flow.py
├── app/                       # 应用核心模块
│   ├── __init__.py
│   ├── clients/               # 第三方服务客户端配置与工具
│   │   ├── __init__.py
│   │   ├── bailian_mcp_config.py
│   │   ├── milvus_config.py         # Milvus 配置
│   │   ├── milvus_utils.py          # Milvus 工具
│   │   ├── minio_config.py          # MinIO 配置
│   │   ├── minio_utils.py           # MinIO 工具
│   │   ├── mongo_history_utils.py   # MongoDB 历史工具
│   │   └── reranker_config.py       # Reranker 配置
│   ├── core/                  # 核心基础设施
│   │   ├── __init__.py
│   │   ├── load_prompt.py         # Prompt 加载器
│   │   └── torch_env.py           # PyTorch 线程限制（防 BGE-M3 段错误）
│   ├── import_process/        # 【导入链路】文件入库处理
│   │   ├── __init__.py
│   │   ├── agent/               # LangGraph 导入 Agent
│   │   │   ├── __init__.py
│   │   │   ├── main_graph.py    # 导入主图（6 节点编排）
│   │   │   └── state.py         # 导入状态定义
│   │   │   └── nodes/           # 导入节点（6 个）
│   │   │       ├── __init__.py
│   │   │       ├── node_bge_embedding.py    # BGE-M3 嵌入
│   │   │       ├── node_document_split.py   # 文档切分
│   │   │       ├── node_entry.py            # 入口节点
│   │   │       ├── node_import_milvus.py    # 导入 Milvus
│   │   │       ├── node_item_name_recognition.py  # 物品名识别
│   │   │       ├── node_md_img.py           # MD 图片提取
│   │   │       └── node_pdf_to_md.py        # PDF→Markdown
│   │   ├── api/                 # 导入 API 服务
│   │   │   ├── __init__.py
│   │   │   └── file_import_service.py  # 导入 API（CORS/路由/后台任务）
│   │   ├── config/              # 导入配置
│   │   │   ├── __init__.py
│   │   │   └── mineru_config.py   # MinerU 配置
│   │   └── page/                # 导入前端页面
│   │       ├── __init__.py
│   │       └── import.html      # 导入管理页面
│   ├── query_process/           # 【问答链路】检索与生成
│   │   ├── __init__.py
│   │   ├── agent/               # LangGraph 问答 Agent
│   │   │   ├── __init__.py
│   │   │   ├── main_graph.py    # 问答主图（5 节点编排）
│   │   │   └── state.py         # 问答状态定义
│   │   │   └── nodes/           # 问答节点（7 个）
│   │   │       ├── __init__.py
│   │   │       ├── node_answer_output.py      # 答案输出
│   │   │       ├── node_item_name_confirm.py  # 物品名确认
│   │   │       ├── node_rerank.py             # Rerank 排序
│   │   │       ├── node_rrf.py                # RRF 融合
│   │   │       ├── node_search_embedding.py   # Embedding 检索
│   │   │       ├── node_search_embedding_hyde.py  # HyDE 检索
│   │   │       └── node_web_search_mcp.py     # 网页搜索 MCP
│   │   ├── api/                 # 问答 API 服务
│   │   │   ├── __init__.py
│   │   │   └── query_service.py        # 问答 API（CORS/路由/SSE）
│   │   └── page/                # 问答前端页面
│   │       ├── __init__.py
│   │       ├── chat.html           # 问答主页面（920行，WorkBuddy 深色风格）
│   │       └── chat_backup_*.html  # 备份文件
│   ├── lm/                      # 语言模型层
│   │   ├── __init__.py
│   │   ├── embedding_config.py  # Embedding 配置
│   │   ├── embedding_utils.py   # Embedding 工具
│   │   ├── llm_utils.py         # LLM 调用工具
│   │   ├── lm_config.py         # LLM 通用配置
│   │   └── reranker_utils.py    # Reranker 工具
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── escape_milvus_string_utils.py
│       ├── normalize_sparse_vector.py
│       ├── rate_limit_utils.py
│       ├── sse_utils.py
│       └── task_utils.py
└── output/                    # 任务产物目录（见下方细分）
    ├── 20261012/              # 按日期分层的上传产物（任务临时文件）
    │   ├── xxx-xxx/backup_chunks.json
    │   └── ...
    ├── architecture_review_2026-10-11.md
    └── redundancy_review_2026-10-12.md
```

---

## 三、核心文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | 导入服务统一入口，复用 `file_import_service.py` 的 FastAPI 实例，监听 8010 端口 |
| `app/import_process/api/file_import_service.py` | 导入 API 核心：文件上传、分流（主文档/附属图片）、后台任务调度、CORS 配置 |
| `app/query_process/api/query_service.py` | 问答 API 核心：`/query`（POST）、SSE 流式、会话管理、健康检查，CORS 配置 |
| `app/import_process/agent/main_graph.py` | 导入链路 LangGraph 主图（BGE-M3 嵌入 → 文档切分 → PDF转MD → 图片提取 → 物品名识别 → Milvus导入） |
| `app/query_process/agent/main_graph.py` | 问答链路 LangGraph 主图（HyDE/HyDE检索 → Embedding检索 → RRF融合 → Rerank → 物品名确认 → 答案输出） |
| `app/chat.html` | 问答前端页面，WorkBuddy 深色风格，920 行，支持 SSE 流式、主题切换、文件上传 |
| `app/import_process/page/import.html` | 导入管理页面 |
| `docker-compose.yml` | Docker Compose 编排（MongoDB 27017、Milvus 19530、Etcd 2379、MinIO 9000/9001、Attu 8000） |
| `Dockerfile` | 容器镜像构建，基于 Ubuntu 22.04 |
| `.env` | 环境变量（MongoDB/Milvus/MinIO 地址、LLM API Key 等） |
| `prompts/*.prompt` | 6 个 Prompt 模板，控制各 Agent 节点行为 |

---

## 四、输出目录 `output/`

该目录已存放约 12 个历史任务产物，按日期 `20261012/` 分层，包含：

- 12 个任务目录（UUID 命名），每个含 `backup_chunks.json`
- 测试文档 `.md`（上传功能验证、采样验证、console 检查等）
- 截图 `.png`（`掌柜智库_8011验证.png` 等）
- 图片资产 `images/`（`flow.png` 等）
- 2 份独立报告：`architecture_review_2026-10-11.md`、`redundancy_review_2026-10-12.md`

---

## 五、已排除项目

以下目录/文件因非项目源代码或可再生内容，不在大纲范围内：

- `.venv/` — Python 虚拟环境
- `__pycache__/` — Python 字节码缓存
- `.git/` — Git 仓库元数据
- `log/` 和 `*.log` — 日志文件
- `*.pyc` — 编译字节码
- `*_bak` / `~$*` — 临时备份文件

---

*大纲生成完毕，可按此结构对代码进行审计或迁移。*
*（内容由AI生成，仅供参考）*
