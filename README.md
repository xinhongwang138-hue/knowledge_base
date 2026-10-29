# 掌柜智库 · 企业级 RAG 知识库

把非结构化的 PDF / Markdown 变成可检索、可溯源、可租户隔离的知识库：文档经 MinerU 解析、
VLM 补图、按句切分后写入 Milvus（稠密 + 稀疏混合检索），问答走「主体确认 → 双路召回 →
RRF 融合 → 断崖截断精排 → 按需联网兜底 → 带引用的流式答案」这一条 LangGraph 状态图。

- **导入服务**：`app/import_process/`，默认 `127.0.0.1:8010`，页面 `/import.html`
- **查询服务**：`app/query_process/`，默认 `127.0.0.1:8011`，页面 `/chat.html`

---

## 1. 能力清单

**数据导入**

| 能力 | 实现要点 |
|---|---|
| 多格式解析 | PDF 走 MinerU 云端 API 转 Markdown（公式/表格/图片）；Markdown 直读 |
| 图文语义对齐 | VLM 为文档图片生成中文摘要，写入切片，检索时图片随答案返回（预签名直链） |
| 智能切片 | 按句边界递归切分，默认 200 字 / 20 字重叠，标题随切片落库（`title` / `parent_title`） |
| 主体识别 | LLM 识别文档主体名（商品/型号），作为检索过滤与幂等清理依据 |
| 混合向量化 | BGE-M3 同时产出稠密（1024 维）+ 稀疏向量，双通道写入 Milvus |
| 幂等入库 | 入库前按 `doc_id`（无则 `item_name`）删除旧切片，同文档重传不产生重复 |
| 批量上传 | 支持一次多文件 + Markdown 附带图片（图片落到 md 同级 `images/`） |

**检索问答**

| 能力 | 实现要点 |
|---|---|
| 主体确认 | 问题里的主体有歧义时直接反问，不让 LLM 猜；同轮识别出 `item_names` 与改写问题 |
| 多轮追问 | 识别「怎么说 / 什么意思」类追问，复用上一轮主体，跳过重复抽取 |
| 双路召回 | 稠密检索 + HyDE（假设答案）检索并发执行，LangGraph 天然同步栅栏 |
| 多路融合 | RRF 按排名融合，规避不同分数量纲不可比的问题 |
| 精排截断 | bge-reranker 交叉编码器精排 + 断崖检测动态截断（相对 25% / 绝对 0.5） |
| 按需联网 | 精排 Top1 低于 `WEB_FALLBACK_MIN_SCORE`（默认 0.5）才触发 MCP 联网，且只触发一次 |
| 流式输出 | SSE 推送 `progress` / `delta` / `final`，前端打字机效果 + 节点进度面板 |
| 来源溯源 | 答案正文 `[i]` 与 `sources` 序号对齐，含文件、章节、chunk、分数、片段，可点击定位 |
| 反馈闭环 | 每条回答可 👍/👎 + 文字评论，落 MongoDB `chat_feedback`，`GET /feedback/stats` 出好评率与差评明细 |

**工程与安全**

- **租户隔离**：`tenant_id` 作为 Milvus 分区键，所有读写强制 `tenant_id == "..."` 过滤；
  租户只能来自 `X-KB-Internal` 内部令牌（HMAC 验签），请求体自报的租户字段一律不采信。
- **写接口保护**：`Origin/Referer` 必须是本机回环地址（防 CSRF / DNS rebinding），
  可选 `X-API-Key`（`WRITE_API_KEY`）二次校验。
- **失败关闭**：`KB_REQUIRE_TENANT=true` 时缺令牌直接 401，绝不退化为全库检索。
- **显存管理**：BGE-M3 与 reranker 互斥加载，切换前 offload + `gc.collect()` + `empty_cache()`；
  4 GB 显存卡上也能跑。
- **MKL 线程限制**：`app/core/torch_env.py` 必须在任何 torch 初始化前导入，否则 BGE-M3 会 segfault。
- **提示词外置**：全部模板在 `prompts/*.prompt`，改文案不用改代码。

---

## 2. 架构

```
┌──────────────────────────────────────────────────────────┐
│ 前端          import.html（导入）   chat.html（问答·SSE）  │
├──────────────────────────────────────────────────────────┤
│ 服务层        FastAPI 导入服务 :8010 │ 查询服务 :8011      │
│              租户令牌 / 来源校验 / 限流 / 健康检查          │
├──────────────────────────────────────────────────────────┤
│ 编排层        LangGraph 导入图（7 节点）│ 查询图（9 节点）   │
├──────────────────────────────────────────────────────────┤
│ 能力层        Qwen（理解/生成） · BGE-M3（嵌入）            │
│               bge-reranker（精排） · MinerU（PDF）         │
├──────────────────────────────────────────────────────────┤
│ 存储层        Milvus（向量·混合检索） MongoDB（历史/反馈）   │
│               MinIO（原文与图片）  etcd（Milvus 元数据）    │
└──────────────────────────────────────────────────────────┘
```

**导入链路**：`entry →(PDF) pdf_to_md → md_img → document_split → item_name_recognition
→ bge_embedding → import_milvus`

**检索链路**：`item_name_confirm →(双路并发) search_embedding / search_embedding_hyde
→ rrf → rerank →(分数不足) web_fallback → answer_output`

---

## 3. 目录结构

```
knowledge_base/
├── app/
│   ├── import_process/
│   │   ├── agent/            # 导入图：main_graph / state / nodes(7)
│   │   ├── api/              # file_import_service.py（:8010，/upload、/status）
│   │   └── page/import.html
│   ├── query_process/
│   │   ├── agent/            # 查询图：main_graph / state / nodes(7)
│   │   ├── api/              # query_service.py（:8011，/query、/stream、/feedback…）
│   │   └── page/chat.html
│   ├── clients/              # Milvus / MinIO / Mongo 客户端与配置
│   ├── lm/                   # LLM / BGE-M3 / reranker 封装（含显存腾挪）
│   ├── core/                 # 日志、提示词加载、torch 环境
│   └── utils/                # 租户、来源校验、SSE、限流、任务状态
├── prompts/                  # 6 个提示词模板（.prompt）
├── eval/                     # 离线评估：qa.csv + eval.py
├── scripts/                  # 租户化重建等运维脚本
├── doc/                      # 测试用 PDF
├── main.py                   # 导入服务统一入口（python main.py）
├── docker-compose.example.yml# 编排模板（复制为 docker-compose.yml 后填口令）
├── .env.example              # 配置模板（复制为 .env）
└── pyproject.toml            # 依赖（uv 管理）
```

---

## 4. 快速开始

### 4.1 前置

- Python 3.11（`uv` 管理依赖，`.venv` 内没有 pip）
- Docker + Docker Compose
- 阿里云百炼 API Key（LLM / VLM）、MinerU Token
- BGE-M3 与 reranker 权重（默认读 `D:/ai_models/modelscope_cache`，可用 `BGE_M3_PATH` 覆盖）

### 4.2 配置

```bash
cp .env.example .env          # 填 API Key / 数据库口令
cp docker-compose.example.yml docker-compose.yml   # 填 Mongo 口令（该文件已被 .gitignore 忽略）
uv sync
```

关键配置项：

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_API_BASE` | DashScope 兼容 OpenAI 接口 |
| `LLM_DEFAULT_MODEL` / `VL_MODEL` | 生成模型 / 图片摘要模型 |
| `MINERU_API_TOKEN` | PDF 解析服务令牌 |
| `MILVUS_URL` / `CHUNKS_COLLECTION` | 向量库地址 / 切片集合名（本地已迁移为 `kb_chunks_v2`） |
| `MONGO_URL` / `MONGO_DB_NAME` | 历史与反馈存储 |
| `MINIO_*` | 原文与图片存储（`MINIO_PUBLIC_ENDPOINT` 决定预签名 URL 的对外地址） |
| `BGE_M3_PATH` / `BGE_RERANKER_PATH` | 本地权重路径（示例配 large，本机实际用 base） |
| `WRITE_API_KEY` | 写接口二次校验口令（留空则不校验 Key） |
| `KB_REQUIRE_TENANT` / `KB_INTERNAL_SECRET` | 租户强制模式与内部令牌密钥（与电商后端保持一致） |
| `WEB_FALLBACK_ENABLED` / `WEB_FALLBACK_MIN_SCORE` | 联网兜底开关与阈值 |
| `MONGO_HISTORY_ENABLED` | 关掉后历史与反馈全部空转（离线评估用，默认开启） |

### 4.3 启动

**容器化（推荐）**

```bash
docker compose up -d          # etcd + MinIO + Milvus + Attu + Mongo + 导入/查询服务
docker compose ps             # import-api / query-api 应为 healthy
```

端口：`8010` 导入 · `8011` 查询 · `8000` Attu · `19530` Milvus · `9000/9001` MinIO · `27018` Mongo。
代码以 bind mount 方式挂载，改完 `docker compose restart query-api` 即生效，无需重建镜像。

**本机直跑**

```bash
python main.py                                              # 导入服务 :8010
python -m app.query_process.api.query_service               # 查询服务 :8011
python main.py --host 0.0.0.0 --port 8010                   # 需要局域网访问时
```

> 本机直跑会加载 BGE-M3 与 reranker，4 GB 显存下不要与容器里的查询服务同时跑，否则 OOM。

---

## 5. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/import.html`（:8010） | 导入页面 |
| POST | `/upload`（:8010） | 多文件上传，触发导入图（支持 `tenant` 由令牌决定） |
| GET | `/status/{task_id}`（:8010） | 导入任务进度（节点级） |
| GET | `/chat.html`（:8011） | 问答页面 |
| POST | `/query`（:8011） | 提交问题；`is_stream=false` 同步返回答案与来源 |
| GET | `/stream/{session_id}`（:8011） | SSE：`ready` / `progress` / `delta` / `final` / `error` |
| GET | `/history/{session_id}`（:8011） | 会话历史（含来源、图片、反馈状态） |
| DELETE | `/history/{session_id}`（:8011） | 清空会话历史 |
| POST | `/feedback`（:8011） | 回答评价：`{session_id, message_id, feedback: up/down, comment}` |
| GET | `/feedback/stats`（:8011） | 反馈统计：好评率 + 最近差评明细 |
| GET | `/health`（两服务） | Mongo / Milvus 连通性，任一不可用返回 503 |

调用示例（合并部署需带内部令牌）：

```bash
curl -X POST http://127.0.0.1:8011/query \
  -H "Content-Type: application/json" \
  -H "X-KB-Internal: <token>" \
  -d '{"query":"HAK180 用什么电源供电？","session_id":"demo","is_stream":false}'
```

`/query` 的返回里带 `message_id`，前端据此把 👍👎 挂到具体那条回答上：

```bash
curl -X POST http://127.0.0.1:8011/feedback \
  -H "Content-Type: application/json" -H "X-KB-Internal: <token>" \
  -d '{"session_id":"demo","message_id":"<上一步返回>","feedback":"down","comment":"没给电压"}'
```

---

## 6. 租户隔离（合并部署必读）

```
电商后端 ──签 HMAC 令牌(X-KB-Internal)──▶ 智库 resolve_tenant ──▶ tenant_id
                                                              ├─ Milvus 分区键 + 强制过滤
                                                              └─ 反馈/历史记录归属
```

- 令牌格式：`base64url(payload).base64url(HMAC-SHA256(payload, KB_INTERNAL_SECRET))`，
  payload 至少含 `tenant` 与 `exp`；与电商侧 `services/auth.py` 同构，但密钥独立。
- `KB_REQUIRE_TENANT=false`（默认）：独立本机模式，无令牌时落到固定租户 `local`。
- `KB_REQUIRE_TENANT=true`：缺令牌 / 验签失败 / 密钥未配置一律 401/503——**刻意失败关闭**，
  不要为了「先跑起来」关掉它。
- 其余租户来源（请求体、查询参数、表单字段）全部不采信。

---

## 7. 评估

```bash
.venv\Scripts\python.exe eval/eval.py                 # 全量（23 条）
.venv\Scripts\python.exe eval/eval.py --limit 3       # 冒烟
.venv\Scripts\python.exe eval/eval.py --tenant ws_1 --out eval/result.csv
```

- 测试集 `eval/qa.csv`：问题 + 标准答案 + `expect_keywords`（必须出现的关键事实）。
- 指标：**上下文召回**（关键事实是否落在召回片段里）、**答案覆盖**（是否答出来）、Top1 精排分、
  召回条数、是否触发联网、耗时。逐条结果写 `eval/result_*.csv`（utf-8-sig）。
- 结果里有两列专门用于诊断：
  - `clarify`：主体确认节点是否直接反问（「您是想问以下哪个产品」）。这类样本没走检索，
    单独计数、不计入召回均值，否则会把数据治理问题误算成检索质量问题；
  - `source_titles`：本轮实际召回的文件与分数，0 召回时用它区分「没检索到」和「检索到别的文档」。
- 召回按**真正进入 Prompt 的上下文**统计：脚本给 `/query` 传 `debug_context=true`
  （非流式专用开关），服务把精排后的完整正文一并返回。不传这个开关时只能拿到 200 字来源摘要，
  长切片后段的关键词会被漏算，指标会系统性偏低。
- 为什么用关键词而不是 Ragas：LLM 当裁判同一份数据两次跑分数会漂，且要先装一大包依赖；
  关键词命中是确定性的，能稳定对比「改前 / 改后」。代价是只看事实是否出现、不看表达。
- 读数注意：服务端 `WEB_FALLBACK_ENABLED=1` 时低分问题会触发联网兜底，单条可能 90 s 以上，
  且联网内容会混进上下文与答案；要纯净的本地检索评估，先把该开关设为 0 再重启服务。

---

## 8. 已知限制

- 任务进度是**进程内内存态**（`app/utils/task_utils.py`），重启即丢；没有文档列表 / 删除界面，
  删除只能靠重传同名文档触发幂等清理。
- 没有独立用户体系，身份与租户完全依赖上游电商后端签发的令牌。
- 断崖截断阈值（25% / 0.5）与切片长度（200 / 20）是经验值，尚未做网格搜索。
- 评估集只有 23 条，且集中在 HAK180 一份手册上，指标绝对值只能当基线看。
- 联网兜底走 MCP，单次 60–100 s，目前只在本地召回明显不足时触发，前端会有较长等待。
- `app/utils/escape_milvus_string_utils.py` 模块 docstring 里有未转义的反斜杠，
  Python 3.12+ 启动时会有 `SyntaxWarning`，不影响功能。
