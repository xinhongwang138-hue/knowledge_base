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
| 智能切片 | 按标题语义初切 + 句边界递归细分，默认 400 字 / 60 字重叠，标题随切片落库（`title` / `parent_title`） |
| 表格摊平 | MinerU 输出的 HTML 表格先转成 Markdown 表格再切分（原始 HTML 无换行无标点，是不可切分的整块） |
| 主体识别 | LLM 识别文档主体名（商品/型号），并按型号 token 收敛到同租户已有的规范名，避免同一产品多个叫法 |
| 混合向量化 | BGE-M3 同时产出稠密（1024 维）+ 稀疏向量，双通道写入 Milvus |
| 幂等入库 | 入库前按 `doc_id`（无则 `item_name`）删除旧切片；主体名按 `(tenant_id, file_title)` 覆盖，同文档重传不产生重复与别名 |
| 批量上传 | 支持一次多文件 + Markdown 附带图片（图片落到 md 同级 `images/`） |
| 文档管理 | 导入页内嵌「已入库文档」列表（MongoDB 台账 `kb_documents`），可一键真删：Milvus 切片 + MinIO 图片 + 主体名索引 + 台账一起清 |

**检索问答**

| 能力 | 实现要点 |
|---|---|
| 主体确认 | 问题里的主体有歧义时直接反问，不让 LLM 猜；同轮识别出 `item_names` 与改写问题 |
| 多轮追问 | 识别「怎么说 / 什么意思」类追问，复用上一轮主体，跳过重复抽取 |
| 双路召回 | 稠密检索 + HyDE（假设答案）检索并发执行，LangGraph 天然同步栅栏 |
| 多路融合 | RRF 按排名融合，规避不同分数量纲不可比的问题 |
| 精排截断 | bge-reranker 交叉编码器精排（输入补「产品名｜章节名」，见 §7）+ 断崖检测动态截断（相对 25% / 绝对 0.5，且至少保留 3 条） |
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
│   │   ├── api/              # file_import_service.py（:8010，/upload、/status、/documents）
│   │   └── page/import.html
│   ├── query_process/
│   │   ├── agent/            # 查询图：main_graph / state / nodes(7)
│   │   ├── api/              # query_service.py（:8011，/query、/stream、/feedback…）
│   │   └── page/chat.html
│   ├── clients/              # Milvus / MinIO / Mongo 客户端与配置（含文档台账 mongo_document_utils）
│   ├── lm/                   # LLM / BGE-M3 / reranker 封装（含显存腾挪）
│   ├── core/                 # 日志、提示词加载、torch 环境
│   └── utils/                # 租户、来源校验、SSE、限流、任务状态、主体名规范化、表格摊平
├── prompts/                  # 6 个提示词模板（.prompt）
├── eval/                     # 离线评估：qa.csv + eval.py
├── scripts/                  # 运维脚本：租户化重建、存量切片重切（reindex_chunks.py）
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
| GET | `/documents`（:8010） | 已入库文档列表；`?sync=true` 从 Milvus 强制回填台账 |
| DELETE | `/documents/{doc_id}`（:8010） | **真删**：Milvus 切片 + MinIO 图片 + 台账记录 |
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

### 文档管理

导入页 `/import.html` 底部内嵌「已入库文档」，列出当前租户已入库的文档（文件名、切片数、主体名、导入时间），
每条可删除。列表来自 MongoDB 台账 `kb_documents`（`(tenant_id, doc_id)` 唯一），
不是内存任务表，所以**服务重启后依然可见**。

```bash
# 列出（台账为空时自动从 Milvus 回填一次；?sync=true 强制重新对齐）
curl http://127.0.0.1:8010/documents -H "X-KB-Internal: <token>"

# 真删：Milvus 切片 + MinIO 图片(upload-images/{file_title}/) + 主体名索引 + 台账记录
curl -X DELETE http://127.0.0.1:8010/documents/<doc_id> \
  -H "X-KB-Internal: <token>" -H "Origin: http://127.0.0.1:8010"
```

删除是**不可逆**的，前端点按钮会先弹确认框列出将被清掉的切片数。
`doc_id` 与 Milvus 切片的 `doc_id` 同源（`sha1(tenant_id|file_title)`），
所以台账里的 id 一定能删到对应切片。

> 合并部署（`KB_REQUIRE_TENANT=true`）下，独立打开 `import.html` 没有令牌会被 401。
> 由外层控制台嵌入时把令牌放进 URL 片段即可：`import.html#t=<token>`（`#` 不进服务端日志）。

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

### 存量切片重切（改了 `CHUNK_SIZE` 之后必做）

切分参数只影响**新导入**的文档。库里已有的切片不会自动跟着变——历史数据是租户化迁移
脚本 `scripts/kb_rebuild_with_tenant.py` 从旧集合原样搬运的（向量都不重算），
所以「代码写 400、库里躺 2142」这种代码与数据不一致会长期存在，页面上完全看不出来。

```bash
docker cp scripts/reindex_chunks.py kb-import-api:/app/reindex_chunks.py
docker exec kb-import-api python /app/reindex_chunks.py --tenant ws_1 --dry-run   # 先看计划
docker exec kb-import-api python /app/reindex_chunks.py --tenant ws_1            # 执行
```

脚本从 Milvus 读回切片正文 → 按标题重组小节 → 摊平 HTML 表格 → 复用
`node_document_split.step_3_refine_chunks` 重切 → 重算 BGE-M3 向量 → 写新删旧。
复用生产切分函数是刻意的：重切的目的就是让数据和「现在导入一份新文档」的结果一致。

> 脚本默认跳过「已经符合当前粒度」的文档（最长切片 ≤ 1.5×`CHUNK_SIZE`）。
> 切片之间有重叠，重复重切会让切口处文字出现两遍，确需重跑加 `--force`。

### 精排输入为什么带「产品名｜章节名」

手册类切片大量使用「本设备」「此插头」「该有害物质」这类指代，正文里既没有产品名也没有章节名。
Cross-Encoder 读的是「问题 + 这一段」，指代补不全就只能按行文风格打分，结果是**分数倒挂**：

| 问题 | 候选切片 | 补上下文前 | 补上下文后 |
|---|---|---|---|
| 电源插头有什么特殊要求 | 「本设备装有接地的插头。此插头只能插入接地的电源插座中…」← 答案原文 | **0.09** | 0.998 |
| 电源插头有什么特殊要求 | 「![设备电源线安全须知：仅限 AC 220-240V…]」← 图注 | **0.99** | 0.982 |
| 产品有害物质表中铅是什么情况 | 「○/× 表示的限量要求见 GB/T 26572…」← 表注 | **0.03** | 0.987 |

图注是完整自述句，指代残缺的要点列表不是，于是精排整体偏向「读起来通顺」而不是「答得上问题」。

检索侧本来没这个问题：入库时向量是用 `f"商品：{item_name}，介绍：{content}"` 生成的
（见 `node_bge_embedding` / `scripts/reindex_chunks.py`）。真正不齐的是精排——它吃的是裸正文，
和向量侧看到的文本根本不是同一份。`node_rerank.build_rerank_text()` 把这一行补上，两侧对齐。

> 四种拼法实测（两条失败用例的目标切片位次，6 = 落在 Top3 之外）：
> 裸正文 `6/6`、只补章节名 `6/6`、**只补产品名 `2/2`、产品名+章节名 `2/1`**。
> 也就是说产品名是关键那一档（解决「这是谁」），章节名再加一点精度（解决「讲哪一节」）——
> 有害物质那条因此从第 2 名提到第 1 名。只补章节名完全无效：`# 电源线` 帮不到「电源插头」这个问题。
>
> 实现上单独走 `rerank_text` 字段，不污染进 Prompt 与出处摘要的 `text`；打完分即从 state 里丢弃。

### 当前基线（2026-10-29，租户 ws_1，23 条）

| 指标 | 首版 | 修主体名/切片粒度 | 修精排输入 |
|---|---|---|---|
| 上下文召回均值 | 0.611 | 0.928 | **0.986** |
| 答案覆盖均值 | 0.667 | 0.928 | **0.986** |
| 检索与答案双命中 | 10/21 | 21/23 | **22/23** |
| 反问条数 | 5 | 0 | 0 |
| 平均 Top1 精排分 | 0.873 | 0.916 | 0.999 |

唯一未命中的是「产品有害物质表中铅（Pb）是什么情况」，缺 `SJ/T`——**这条缺的不是检索能力**：
干净版源文件里那句话本来就是「本表格依据 的规定编制。」，`SJ/T 11364` 在源 md 就没有，
只存在于另一份带乱码的 PDF 副本里（详见 §8）。`铅` 与 `GB/T 26572` 均已召回到位。

---

## 8. 已知限制

- 任务进度是**进程内内存态**（`app/utils/task_utils.py`），重启即丢（文档列表已改为 MongoDB 台账，不受影响）。
- 没有独立用户体系，身份与租户完全依赖上游电商后端签发的令牌。
- 断崖截断阈值（25% / 0.5）与切片长度（400 / 60）是经验值，尚未做网格搜索；
  下限固定保留 3 条是为了兜住「一条高置信切片把其余全截掉」的单点失真。
- 补了精排上下文后同一产品的切片分数普遍逼近 1（0.98~0.999），断崖更难触发，
  实际效果退化成「保留 RRF 给的全部候选」（上限 10 条）。这是刻意取舍：多带两片同样相关的
  上下文，代价只是几百字 Prompt，而误截的代价是答案直接不在上下文里。
- 评估集只有 23 条，且集中在 HAK180 一份手册上，指标绝对值只能当基线看。
- 联网兜底走 MCP，单次 60–100 s，目前只在本地召回明显不足时触发，前端会有较长等待。
- 同一份 HAK180 手册在库里有**两份互不完整的副本**，且各有缺陷，谁也替不了谁：

  | 文件 | 缺陷 |
  |---|---|
  | `Brother产品手册`（PDF/MinerU） | 个别页文字被识别成乱码；标题层级是 `#`，同一节被切成 7 片 |
  | `hak180产品安全手册`（md） | 干净，但「本表格依据 **SJ/T 11364** 的规定编制」丢成了「本表格依据 的规定编制」 |

  两者逐句比对重叠 83%（整体相似度 0.846），精排后会互占召回位。属于数据治理问题：
  系统没有自动去重（两份文件名、`doc_id`、`item_name` 都不同，任何自动合并都会有误伤）；
  要根治得换一份完整的原始手册重新导入，然后用 `DELETE /documents/{doc_id}` 删掉多余副本。
- PDF 个别页的文字被 MinerU 识别成乱码（实测一段 212 字切片开头约 25 字是马来亚拉姆文区码点的乱码），
  目前不做清洗：无法可靠区分乱码与生僻字，误删风险高于收益。补了精排上下文后它已从第 1 名
  （0.99）掉到第 5 名（0.98 以下），不再挤掉真实内容，但仍在库里。
- `app/utils/escape_milvus_string_utils.py` 模块 docstring 里有未转义的反斜杠，
  Python 3.12+ 启动时会有 `SyntaxWarning`，不影响功能。
