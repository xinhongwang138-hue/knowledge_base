---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 2ef7930562d418b0a12963f90da05d53_e00ddb90c55f11f19b82525400de85a5
    ReservedCode1: mMFpsSJ4YM9WW/11vB2s5+Dx2iRtlt3chq43cCLLUlGCsKeAgPFy8iXB/C6HY8Wi9yx+pFEevs0xSDLZ/Ro+7XfSCCJNst9k55x4H0aqo35Kgm6e6o9fpdKoQp1hPC8UyhpRyQSpwNqTwnSL63LNd/Ij8wI8Hok7/u15JCSARn6C68pdsG97wZwTdYk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 2ef7930562d418b0a12963f90da05d53_e00ddb90c55f11f19b82525400de85a5
    ReservedCode2: mMFpsSJ4YM9WW/11vB2s5+Dx2iRtlt3chq43cCLLUlGCsKeAgPFy8iXB/C6HY8Wi9yx+pFEevs0xSDLZ/Ro+7XfSCCJNst9k55x4H0aqo35Kgm6e6o9fpdKoQp1hPC8UyhpRyQSpwNqTwnSL63LNd/Ij8wI8Hok7/u15JCSARn6C68pdsG97wZwTdYk=
---

# 掌柜智库知识库 全架构审查报告

> 审查日期：2026-10-11
> 审查对象：`D:\PythonProject\knowledge_base`（Python + LangChain + Milvus(BGE-M3) + FastAPI + MongoDB + MinIO 的 RAG 问答系统）
> 审查方式：通读项目目录、导入链路、检索链路（Agent 状态机）、配置与客户端封装全部关键代码
> 说明：本报告基于当前工作副本代码现状撰写；历史记忆中提到的"node_item_name_confirm 硬编码拒识句"在当前代码中已改为"通用检索模式"（见 2.2.1），不再存在。

---

## 1. 架构全景

### 1.1 目录结构

| 目录/文件 | 职责 |
|---|---|
| `app/clients/` | 外部服务客户端封装：Milvus（`milvus_utils.py`/`milvus_config.py`）、MongoDB 历史（`mongo_history_utils.py`）、MinIO 对象存储（`minio_utils.py`/`minio_config.py`）、百炼 MCP 联网搜索（`bailian_mcp_config.py`） |
| `app/conf/` | 通用配置 |
| `app/core/` | 基础支撑：日志（`logger.py`）、提示词加载（`load_prompt.py`）、PyTorch/MKL 环境初始化（`torch_env.py`） |
| `app/import_process/` | **导入链路**：Web 服务（`api/file_import_service.py`）、LangGraph 主图（`graph/main_graph.py`）、状态（`state.py`）、8 个节点（入口/PDF转MD/图片处理/分块/主体识别/向量化/入库） |
| `app/lm/` | 模型层：Embedding（`embedding_utils.py`/`embedding_config.py`）、LLM（`llm_utils.py`/`lm_config.py`）、Reranker（`reranker_utils.py`） |
| `app/query_process/` | **检索链路**：Web 服务（`api/query_service.py`）、Agent 主图（`agent/main_graph.py`）、状态（`agent/state.py`）、8 个节点（主体确认/普通向量检索/HyDE检索/联网搜索/RRF融合/重排/答案输出） |
| `app/utils/` | 通用工具：SSE 推送（`sse_utils.py`）、任务状态（`task_utils.py`）、限速（`rate_limit_utils.py`）、Milvus 字符串转义等 |
| `prompts/` | 6 个提示词模板（`answer_out` / `rewritten_query_and_itemnames` / `hyde_prompt` / `image_summary` / `item_name_recognition` / `product_recognition_system`） |
| `output/`、`doc/`、`test/` | 输出物、文档、测试 |
| `main.py` | 导入服务统一入口（先导入 `torch_env` 限制 MKL 线程，默认 127.0.0.1:8010） |
| `pyproject.toml` / `Dockerfile` / `docker-compose.yml` | 依赖声明 / 镜像构建 / 依赖服务编排（etcd+minio+milvus+attu+mongo+import-api+query-api） |
| `.env` / `.gitignore` | 环境变量（密钥） / 忽略规则（.env/.venv/logs） |

### 1.2 模块职责与数据流

```
【导入链路】 文档入库（建库）
 上传 PDF/MD → node_entry（类型分流）
   → node_pdf_to_md（MinerU 在线转换 PDF→MD）
   → node_md_img（本地图片→多模态总结→传 MinIO→替换 URL 为外链）
   → node_document_split（按标题初切 + RecursiveCharacterTextSplitter，200/20）
   → node_item_name_recognition（LLM 识别商品名→BGE-M3 向量化→写 kb_item_names）
   → node_bge_embedding（"商品名+内容"拼接批量向量化，batch=5）
   → node_import_milvus（稠密+稀疏+元数据入库 kb_chunks，按 item_name 删旧保幂等）
   → END

【检索链路】 问答出库（查库）
 POST /query → node_item_name_confirm（历史读取→LLM 提取商品名+改写问题
   →向量化对齐 kb_item_names→三分支：确认/候选反问/通用模式）
   →（有 answer 则直接输出）→ 三路并发：
       node_search_embedding（BGE-M3 混合检索，item_name 过滤）
       node_search_embedding_hyde（LLM 生成假设文档+问题拼接再检索）
       node_web_search_mcp（百炼 MCP 联网搜索，Streamable HTTP）
   → node_rrf（带权 RRF 融合，k=60，权重 1.0/1.0/0.8）
   → node_rerank（BGE-Reranker 打分 + 动态 TopK 截断）
   → node_answer_output（检查前置答案→组装 Prompt→LLM 流式/非流式生成
     →提取图片 URL→写 MongoDB→推 SSE FINAL）
   → END

【基础设施】
 Milvus 三件套（etcd/minio/standalone）+ Attu（docker-compose）
 MongoDB kb002（历史消息，复合索引 (session_id,ts)）
 MinIO knowledge-base-files（图片公开读）
 BGE-M3（embedding）/ bge-reranker-large（rerank）/ OpenAI 兼容 API（LLM+视觉）/ MinerU（PDF 转换）
```

### 1.3 关键设计亮点

- 主体确认（item_name）先对齐再检索：避免多型号产品串味，是召回精度的核心设计。
- 三路召回（普通向量 + HyDE + 联网）汇入 RRF，异常单路降级为空结果，链路鲁棒。
- 状态 TypedDict + 深拷贝默认值，避免 LangGraph 全局污染。
- `torch_env` 在最早入口导入，规避 Intel MKL 在 E-core 上的段错误（MKL_NUM_THREADS=1，且不设 OMP）。
- 导入链路幂等（按 item_name 删旧插新）、限速（滑动窗口）、多模态图片替换 MinIO 外链。
- SSE 队列解耦后台跑图与前端消费，支持打字机流式。

---

## 2. 各链路关键实现说明

### 2.1 导入链路

| 节点 | 关键实现 | 参数/行为 |
|---|---|---|
| `node_entry.py` | 生成 task_id、校验后缀（.pdf/.md/.markdown），按类型设置开关 | 不支持类型仅告警返回 |
| `node_pdf_to_md.py` | MinerU API 上传→轮询→下载→解压→定位重命名 md | `trust_env=False` 防代理签名；`_sanitize_name` 净化文件名；<200MB |
| `node_md_img.py` | 扫本地图→多模态总结→传 MinIO→替换 URL→备份 `_new.md` | `apply_api_rate_limit` 限速；`CONTEXT_CHARS=100` |
| `node_document_split.py` | 按标题初切（跳过代码块）+ RecursiveCharacterTextSplitter | `CHUNK_SIZE=200`、`OVERLAP=20`；备份 `backup_chunks.json` |
| `node_item_name_recognition.py` | LLM 识别 item_name 回填 chunks，BGE-M3 向量化后写 `kb_item_names` | `LLM_MAX_RETRY=2`；幂等删旧插新；`CONTEXT_TOTAL_MAX_CHARS=2500` |
| `node_bge_embedding.py` | 「商品名+内容」拼接批量向量化 | `EMBED_BATCH_SIZE=5`；异常批次保留原数据；单例模型 |
| `node_import_milvus.py` | 稠密+稀疏+元数据入库 `kb_chunks`，按 item_name 删旧 | `chunk_id INT64` 主键；插入回填 chunk_id |
| `file_import_service.py` | POST /upload 多文件后台导入；GET /status/{task_id} 轮询；GET /import.html | 端口 8010；`ALLOWED_SUFFIXES`；`MAX_FILE_SIZE_MB=100`；CORS 放开 |

### 2.2 检索链路

#### 2.2.1 主体确认节点 `node_item_name_confirm.py`

七步流程：取历史 → 存用户问题 → LLM 提取 `item_names` + 改写问题（`json_mode=True`）→ 商品名向量化在 `kb_item_names` 混合检索（权重 0.8/0.2）→ 评分对齐（>0.85 确认 / >=0.6 候选 / <0.6 丢弃）→ 三分支 → 写回历史。

三分支：
- **分支A（确认）**：`item_names` 写入 state，回填历史中缺商品名的消息，`has_product=True`。
- **分支B（候选）**：生成反问语写入 `answer`（"您是想问以下哪个产品：XXX？"），主图路由直接进答案输出，跳过检索。
- **分支C（无结果）**：**当前版本已不再硬编码拒识句**，而是 `has_product=False` 走通用检索模式（代码注释明确"不再拦截"）。

#### 2.2.2 三路召回

| 节点 | 实现 | 关键参数 |
|---|---|---|
| `node_search_embedding.py` | BGE-M3 混合检索（稠密 0.8/稀疏 0.2），`item_name in [...]` 过滤 | `RETRIEVE_TOP_K=5`、`REQUEST_LIMIT=10`；`is_general`（`has_product is False`）时跳过过滤走全库；item_names 空且非通用时直接返回空 |
| `node_search_embedding_hyde.py` | LLM 生成假设文档（`hyde_prompt`），与改写问题拼接后再混合检索 | HyDE 失败降级为空；item_names 空时**直接全库检索**（与普通路行为不一致，见问题 P1-4） |
| `node_web_search_mcp.py` | openai-agents SDK + `MCPServerStreamableHttp` 调百炼 `bailian_web_search` | `asyncio.run` 桥接同步节点；`MCP_TIMEOUT=300`；未配置时优雅降级为空 |

#### 2.2.3 融合与重排

- `node_rrf.py`：带权 RRF 融合（k=60；embedding/hyde/web 权重 1.0/1.0/0.8），兼容 Milvus Hit/嵌套字典/扁平字典三种形态，联网结果转 `text` 字段，跨源去重取 Top-10。
- `node_rerank.py`：合并本地 RRF 切片 + 联网结果，BGE-Reranker（transformers 加载，`use_safetensors=False`）打分，动态 TopK 截断（`RERANK_MAX_TOPK=10`、`MIN=1`、`GAP_RATIO=0.25`、`GAP_ABS=0.5`）；异常降级原始顺序。

#### 2.2.4 答案生成 `node_answer_output.py`

1. 检查前置 answer（分支 B/C 产物）→ 有则直接 SSE 推送，不调 LLM；
2. 组装 Prompt（`answer_out.prompt`）：Rerank 后文档按序拼接（元数据标签：序号/来源/chunk_id/score/title），`MAX_CONTEXT_CHARS=12000` 截断；
3. LLM 生成：流式逐字推送 DELTA；非流式 `invoke` 后 `set_task_result`；
4. 从文档正文正则提取 Markdown 图片 URL（校验后缀）补进答案；
5. 写 MongoDB（assistant 消息带 `image_urls`）+ 推送 FINAL。

### 2.3 支撑组件

| 组件 | 实现要点 |
|---|---|
| `sse_utils.py` | 进程内 `session_id -> asyncio.Queue`；事件类型 READY/PROGRESS/DELTA/FINAL/ERROR；队列满丢弃最旧事件；`sse_generator` 心跳 15s |
| `task_utils.py` | 进程内内存追踪：`_TASK_GLOBAL_STATUS`/`_TASK_RESULT`/节点记录；导入与查询**共用**此命名空间（见问题 P1-3） |
| `milvus_utils.py` | 单例客户端；`create_hybrid_search_requests` + WeightedRanker；失败返回 None/空列表由调用方降级 |
| `mongo_history_utils.py` | PyMongo 懒连接单例；`save_chat_message` 增改；`get_recent_messages` 复合索引取最近 N 条正序 |
| `minio_utils.py` | 懒加载单例；桶不存在则创建并设**公开读策略（Principal "\*"）**（见问题 P1-5） |
| `reranker_utils.py` | FlagReranker 段错误→改 transformers 加载；`_RERANK_LOAD_LOCK`+`_RERANK_LOCK` 防并发段错误；`RERANKER_RELEASE_AFTER_SCORE` 可释放（见问题 P2-1） |
| `llm_utils.py` | ChatOpenAI 全局缓存（provider/model/json_mode 三键）；双 provider（文本 OPENAI_API_*、视觉 VL_API_* 回退主密钥）；`temperature=0.1` |
| `embedding_utils.py` | BGE-M3 单例；`normalize_embeddings=True`；NumPy 转原生类型；不吞异常 |

---

## 3. 问题与风险分级

> 分级：P0 致命/必现 → P1 高 → P2 中 → P3 低

### P0 致命

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P0-1 | **docker-compose.yml 卷定义语法错误**：顶层 `volumes:` 段三个 bind 卷被写成 ` n`（前导空格 + 字母 n），而服务引用的卷名是 `etcd`/`minio`/`milvus`，顶层并未定义 | `docker-compose.yml` 第 167-183 行 | `docker compose up -d` 报卷未定义错误，或创建匿名卷导致 Milvus 数据不落盘；另有 `- milvus/milvus` 短语法（卷名+子路径语义）与 bind 卷意图不符 |
| P0-2 | **明文凭据进 compose 文件**：`MONGO_URL` 硬编码 `mongodb://admin:Wxh15987456@mongo:27017/...`，且 `docker-compose.yml` 未被 `.gitignore` 覆盖 | `docker-compose.yml` 第 107/141 行 | 一旦提交仓库即泄露数据库口令；compose 文件属于会进入版本库的文件 |

### P1 高

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P1-1 | **SSE 会话队列内存泄漏**：`_SESSIONS` 全局 dict 只在客户端断开后由 `sse_generator` break 退出，但生成器**没有调用 `close_session`**；POST /query 每次新建队列也不清理；队列无 TTL | `app/utils/sse_utils.py`、`app/query_process/api/query_service.py` | 长期运行后 session 与队列只增不减，内存持续上涨；重启后所有进行中会话丢失 |
| P1-2 | **任务状态共用内存命名空间且无持久化**：导入 task_id 与查询 session_id 共用 `_TASK_GLOBAL_STATUS`/`_TASK_RESULT`；重启丢失全部状态 | `app/utils/task_utils.py` | ID 碰撞导致状态串扰（导入任务状态显示到查询进度）；服务重启后前端轮询 /status 拿不到结果 |
| P1-3 | **同步阻塞链路无并发控制**：三路检索内 Embedding/Retrieval/Rerank 全同步，MCP 用 `asyncio.run` 桥接；FastAPI 默认线程池（~40）容易被单个请求的多次阻塞调用耗尽 | 各检索节点、`node_rerank.py`、`node_web_search_mcp.py` | 高并发下请求排队、SSE 推送延迟；MCP 每次调用都新建事件循环 + connect/cleanup，开销大 |
| P1-4 | **两路向量检索对"无商品名"行为不一致**：`node_search_embedding` 在 item_names 空且非通用模式时返回空，而 `node_search_embedding_hyde` 空 item_names 时直接全库检索；通用模式（`has_product=False`）下两路都全库，召回易跨产品串味 | `node_search_embedding.py` L60-63、`node_search_embedding_hyde.py` step_2 | 通用问题（如"怎么保养"）可能召回多个产品切片，LLM 答案混杂；两路行为不一致使 RRF 权重失衡 |
| P1-5 | **MinIO 桶公开读**：桶创建即设 `Principal "*"` 公开读策略，endpoint 为公网 IP | `app/clients/minio_utils.py` | 任何知道图片 URL 的人可读取全部上传图片；若涉产品内部资料则为数据泄露 |
| P1-6 | **LLM JSON 提取无重试**：`step_3_extract_info` 一次 `json.loads` 失败即降级为空 item_names（无重试、无校验），而导入链路 `node_item_name_recognition` 有 `LLM_MAX_RETRY=2` | `node_item_name_confirm.py` step_3 | 模型偶发返回非 JSON → 明确商品名被丢弃 → 走通用模式召回精度下降，且无告警 |

### P2 中

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P2-1 | **Reranker 释放策略/吞吐**：`RERANKER_RELEASE_AFTER_SCORE` 开启时每次检索完释放模型，下个请求重新加载数百 MB 权重；即使常驻，CPU（MKL 单线程）推理慢 | `app/lm/reranker_utils.py` | 吞吐瓶颈；重排延迟叠加在检索链路末端 |
| P2-2 | **Milvus 删旧按 item_name 全量收集**：`node_import_milvus` 先收集集合中所有 item_name 再构造删除 expr；若 item_name 与历史版本不完全一致（如"华为P60" vs "华为P60 128G"），旧切片成孤儿数据，无法被过滤召回、无法发现 | `app/import_process/nodes/node_import_milvus.py` | 数据孤儿累积；expr 过长风险 |
| P2-3 | **导入任务可靠性**：MinerU 轮询无最大次数上限（仅有轮询间隔配置），任务状态仅内存 | `node_pdf_to_md.py`、`file_import_service.py` | 外部服务卡死则任务永久 pending；重启丢进度 |
| P2-4 | **CORS 全放开 + 凭据**：`allow_origin_regex=".*"` + `allow_credentials=True` | `file_import_service.py`、`query_service.py` | 任意站点可携带凭据跨域调用，CSRF/接口滥用风险 |
| P2-5 | **`has_product` 默认值语义风险**：默认状态 `has_product=True`（与"默认无商品"直觉相反）；若异常路径未走 step_6，会触发"空 item_names + 非通用 → 检索直接返回空" | `app/query_process/agent/state.py` L49 | 极端异常下整个检索空转，LLM 无参考内容 |
| P2-6 | **prompts 缺"无参考内容"兜底策略**：`answer_out.prompt` 只说"不要编造"，未定义上下文为空/完全无关时的明确行为；`step_2_construct_prompt` 在 docs 为空时写"无参考内容" | `prompts/answer_out.prompt`、`node_answer_output.py` | LLM 可能硬凑答案或答非所问，无确定性兜底 |
| P2-7 | **多模态图片处理串行**：`node_md_img` 每图一次多模态调用 + 上传，限速串行，大文档图片多时导入极慢；无失败重试/断点 | `app/import_process/nodes/node_md_img.py` | 导入吞吐低 |
| P2-8 | **日志命名冲突风险**：`logger.py` 注释提示内部 logger 定义与末尾全局 logger 命名冲突 | `app/core/logger.py` | 潜在模块级变量覆盖，影响日志输出正确性（需复核） |

### P3 低

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P3-1 | RRF k=60、权重 1.0/1.0/0.8 硬编码在调用处 | `node_rrf.py` | 调参需改代码 |
| P3-2 | MCP 每次 `asyncio.run` + connect/cleanup，无连接池 | `node_web_search_mcp.py` | 每次联网多 2 次网络握手 |
| P3-3 | 候选反问（分支 B）不可点击选择，用户需手输型号 | `node_item_name_confirm.py` + 前端 `chat.html` | 交互体验差 |
| P3-4 | 依赖双份维护：`pyproject.toml` 与 `Dockerfile` 各列一份，易漂移；镜像含 magic-pdf + torch CPU 全量依赖，体积大 | `pyproject.toml`、`Dockerfile` | 依赖不一致风险、镜像臃肿 |
| P3-5 | 检索/导入无结构化指标日志（召回条数、耗时、分数分布、各节点时延） | 各节点 | 调优靠猜，无法定位瓶颈 |
| P3-6 | import-api/query-api 环境变量重复硬编码（.env ro 挂载 + 容器 env 覆盖双份） | `docker-compose.yml` | 配置漂移 |

---

## 4. 改进方案

### P0 修复（建议立即执行）

**P0-1 docker-compose 卷修复**
文件：`docker-compose.yml` 第 167-183 行
```yaml
volumes:
  etcd:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: C:\Users\W\WorkBuddy\2026-09-27-21-25-54\milvus\volumes\etcd
  minio:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: C:\Users\W\WorkBuddy\2026-09-27-21-25-54\milvus\volumes\minio
  milvus:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: C:\Users\W\WorkBuddy\2026-09-27-21-25-54\milvus\volumes\milvus
```
同时修正 standalone 挂载为 `- milvus:/var/lib/milvus`（Milvus 数据目录，避免挂到不生效的 /milvus）；修复后执行 `docker compose config -q` 校验。

**P0-2 凭据外置**
文件：`docker-compose.yml`、`.gitignore`
- compose 内 `MONGO_URL` 改为 `${MONGO_URL}` 引用；
- `.gitignore` 追加 `docker-compose.yml`（或提交一个 `docker-compose.example.yml` 脱敏模板）；
- `.env` 继续严格保密（不入库、不随备份外发）。

### P1 改进

**P1-1 SSE 队列生命周期管理**
文件：`app/utils/sse_utils.py`、`app/query_process/api/query_service.py`
- `sse_generator` 改用 `try/finally`，客户端断开或 FINAL 后调 `close_session(session_id)`；
- 队列加 TTL：`push_to_session` 前检查队列创建时间，超时（如 10 分钟）自动重建；
- 后台跑图完成（FINAL 已推送）后延迟清理队列，避免事件堆积。

**P1-2 任务状态隔离与持久化**
文件：`app/utils/task_utils.py`
- 为导入与查询加命名空间前缀（如 `import:{task_id}` / `query:{session_id}`）；
- 节点记录改为落 MongoDB（复用 `kb002`）或 SQLite，重启可恢复；
- 保留内存态作热缓存，加 TTL 清理。

**P1-3 检索链路并发化**
文件：`node_search_embedding.py`、`node_search_embedding_hyde.py`、`node_web_search_mcp.py`、`node_rerank.py`
- LangGraph 节点改 async（`async def` + `await`），Embedding/Rerank 用 `asyncio.to_thread` 隔离，避免阻塞事件循环；
- MCP 客户端做模块级连接池（一次 connect 多次 call_tool，复用会话），去掉每次 `asyncio.run`；
- 或最小改动：三路检索放入独立线程池（`ThreadPoolExecutor(max_workers=3)`）并行执行，保持同步节点形态。

**P1-4 统一"无商品名"检索语义**
文件：`node_search_embedding.py`、`node_search_embedding_hyde.py`
- 两路统一：`item_names` 为空时都走全库（或都跳过，二选一），行为一致；
- 通用模式加"产品相关性二次过滤"：RRF 后按 chunk 的 `item_name` 聚类，仅保留与问题中实体词（从 `rewritten_query` 提取）相关的组，或把"跨产品混合"作为提示词约束（`answer_out.prompt` 增加"仅采用与问题直接相关的内容，忽略无关产品信息"）。

**P1-5 MinIO 私有化**
文件：`app/clients/minio_utils.py`、`node_answer_output.py`、前端
- 桶策略改为私有，图片 URL 改为预签名 URL（MinIO `presigned_get_object`，有效期 1h）；
- 或最小改动：保留公开读但加 referer 白名单策略。

**P1-6 LLM 提取重试与校验**
文件：`node_item_name_confirm.py` step_3
- 参照导入链路：`LLM_MAX_RETRY=2`，`json.loads` 失败重试；
- 校验：提取的 `item_names` 应出现在查询或历史文本中（宽松子串匹配），否则丢弃并告警；
- 增加结构化日志：提取失败次数、降级原因。

### P2 改进

**P2-1 Reranker 常驻与加速**：文件 `app/lm/reranker_utils.py`——默认不释放（关闭 `RERANKER_RELEASE_AFTER_SCORE` 或仅空闲 5 分钟释放）；尝试 ONNX/量化（`optimum`）或缩小 `REQUEST_LIMIT` 减少待排文档数。

**P2-2 文件级幂等**：文件 `node_import_milvus.py`——为 schema 增加 `doc_id`（文件名哈希），删旧按 `doc_id` 而非仅 `item_name`；孤儿数据可定期扫描 `item_name` 分布审计。

**P2-3 导入任务超时与断点**：文件 `node_pdf_to_md.py`、`file_import_service.py`——MinerU 轮询加最大次数/总时长上限（如 10 分钟），超时标记 failed；任务状态落库后支持断点续跑。

**P2-4 CORS 白名单**：文件 `file_import_service.py`、`query_service.py`——`allow_origins` 改为显式列表（本地开发 `http://127.0.0.1:*` + 生产域名），去除 `.*` 正则。

**P2-5 默认值改安全向**：文件 `app/query_process/agent/state.py`——`has_product` 默认改为 `False`（通用模式全库召回优于空召回），并在 `node_search_embedding` 对"空 item_names 且 has_product 缺失"显式告警。

**P2-6 提示词兜底**：文件 `prompts/answer_out.prompt`——增加："若【参考内容】为空或与问题无关，请明确回复'知识库中暂无相关资料，建议联网搜索或补充文档'，不要编造。"

**P2-7 图片处理并发与重试**：文件 `node_md_img.py`——多图并发（`ThreadPoolExecutor` 上限 3）+ 单图失败重试 2 次；已处理图片按哈希幂等跳过。

**P2-8 logger 命名复核**：文件 `app/core/logger.py`——确认内部 logger 与全局 logger 是否同名冲突，统一为 `logger = loguru.logger` 单一出口。

### P3 可选优化

- **P3-1**：`node_rrf.py` 权重/参数移入 `app/conf/` 或 `.env`。
- **P3-2**：`node_web_search_mcp.py` 做 MCP 会话池。
- **P3-3**：前端 `chat.html` 对分支 B 的 answer 渲染为可点击选项按钮，点击后带 `item_names` 重发。
- **P3-4**：依赖单一来源（`pyproject.toml` + `uv lock`），Dockerfile 用 `uv pip install --system` 安装锁定依赖；import/query 拆分为更轻镜像或保留共用但裁剪 magic-pdf 相关（query 容器不需要）。
- **P3-5**：各节点 `@node_log` 输出结构化 JSON（节点名/耗时/召回数/分数分布），可接入简单指标聚合。
- **P3-6**：compose 环境变量收敛到 `.env` 单一来源，容器 env 仅保留必要的链路内地址。

---

## 5. 结论

系统整体架构清晰：主体确认 + 三路召回 + RRF/重排 + LLM 生成的分层设计合理，异常降级与幂等处理扎实，属于可运行的成熟 RAG 骨架。当前**最紧迫**的是 P0-1（compose 卷语法错误，直接导致编排不可用）与 P0-2（明文凭据），其次是 SSE 内存泄漏（P1-1）、任务状态命名空间与持久化（P1-2）、并发与无商品名检索语义统一（P1-3/P1-4）。建议按 P0 → P1 → P2 顺序落地，P0 与 P1 全部完成后即可支撑稳定的联调与生产部署。
*（内容由AI生成，仅供参考）*
