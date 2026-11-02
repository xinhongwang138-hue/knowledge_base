# 企业知识库（Knowledge Base · RAG）

把散在网盘里的 PDF 变成能问答的知识库。两条 LangGraph 状态机：一条负责**入库**（解析 → 切图 → 分块 → 向量化 → 落库），一条负责**查询**（双路召回 → RRF 融合 → 重排 → 命中不足时联网兜底）。

节点是独立的函数，流程可单独重放：换解析器只动 `node_pdf_to_md`，换向量模型只动 `node_bge_embedding`，其余不用碰。

---

## 一、两条链路

### 1. 入库（`kb_import_app`）

```
node_entry              判断文件类型
 ├─(PDF) → node_pdf_to_md       MinerU 解析成 Markdown
 ├─(MD)  → node_md_img          直接进入图片处理
 └─(其它) → END
node_md_img             抽图 → 上传 MinIO → 视觉模型给每张图写摘要
                        → 把 `![](本地图)` 替换成 `![图片摘要](MinIO URL)`
node_document_split     分块
node_item_name_recognition  文本模型从前 5 个分块里认出条目名（品名 / 产品名），
                        再把条目名做稠密+稀疏向量写入 kb_item_names
node_bge_embedding      BGE-M3 向量化
node_import_milvus      写入 Milvus
```

固定连边：`node_pdf_to_md → node_md_img → node_document_split → node_item_name_recognition → node_bge_embedding → node_import_milvus → END`。

两个容易被忽略的设计：

- 图片不是丢弃或只存路径，而是**先让视觉模型写一句摘要、再替换成 `![摘要](URL)`**——这样图片内容也能被检索到，而不只是"原文里有张图"。
- 条目名单独建 `kb_item_names` 集合，是为了让"这个万用表怎么用"能先定位到具体文档，再在文档内部检索。

### 2. 查询（`kb_query_app`）

```
node_item_name_confirm   确认问的是哪个条目
 ├─(能确认) → node_search_embedding        向量检索
 │            node_search_embedding_hyde   HyDE 改写后再检索
 │                └────────┬───────────────┘
 │                    node_rrf            两路结果 RRF 融合（等两路都跑完）
 │                    node_rerank         bge-reranker-large 精排
 │                         ├─(Top1 ≥ 0.5) → node_answer_output
 │                         └─(不足)       → node_web_fallback（百炼 MCP 联网，只跑一次）
 └─(不能确认) → node_answer_output
```

两个细节写在代码里，不是约定：

- 双路检索靠 LangGraph 的**同步栅栏**汇合到 `node_rrf`——两个节点都执行完才会进下一步，不需要手写 join。
- 联网兜底的阈值是 `WEB_FALLBACK_MIN_SCORE`（默认 `0.5`），有本地片段且分数达标就**不**联网，避免多花一次检索的钱。

---

## 二、技术栈

| 层 | 选型 |
| --- | --- |
| 编排 | LangGraph 1.x（两条独立 Graph，共享 state 定义） |
| 服务 | FastAPI + uvicorn（入库 8010 / 查询 8011） |
| 向量 | BGE-M3（`flagembedding`）→ Milvus 3.0 |
| 精排 | BAAI/bge-reranker-large |
| 模型调用 | OpenAI 兼容接口（默认阿里云百炼）：文本 `qwen-plus`，视觉 `qwen-vl-max` |
| 解析 | MinerU（`magic-pdf`） |
| 存储 | Milvus（向量）· MinIO（原图）· MongoDB（历史会话） |
| 联网 | 百炼 MCP（`MCP_DASHSCOPE_*`） |
| 依赖管理 | uv（`uv.lock`）+ `requirements.lock.txt`，Python ≥ 3.11 |

Milvus 里实际在写的两个集合：`kb_chunks`（分块向量）、`kb_item_names`（条目名，稠密+稀疏）。代码里还定义了第三个 `kb_graph_entity_names`（`.env` 中叫 `kb_entity_names`），**当前没有任何节点写入它**——是预留位，不是功能。

---

## 三、目录结构

```
.
├─ main.py                        统一入口：python main.py 起入库服务（8010）
├─ docker-compose.example.yml     7 个服务：etcd / minio / milvus / attu / mongo / import-api / query-api
├─ Dockerfile
├─ start_all.ps1 / start_all.bat  本机一键启动（注意其中的 $PROJECT 是绝对路径，见「已知边界」）
├─ .env.example                   环境变量模板
├─ app/
│  ├─ core/                       torch_env（线程数锁定）、logger、load_prompt
│  ├─ clients/                    milvus / minio / mongo / reranker / 百炼 MCP 的封装
│  ├─ lm/                         模型配置与 embedding、reranker、LLM 调用
│  ├─ import_process/
│  │  ├─ agent/main_graph.py      入库 Graph
│  │  ├─ agent/nodes/             7 个节点，一节点一文件
│  │  ├─ api/file_import_service.py
│  │  └─ page/import.html         上传页面
│  └─ query_process/
│     ├─ agent/main_graph.py      查询 Graph
│     ├─ agent/nodes/             7 个节点（另有 1 个 MCP 检索辅助模块）
│     ├─ api/query_service.py
│     └─ page/chat.html           问答页面
├─ prompts/                       各节点提示词
├─ test/04-test_graph_flow.py     唯一的链路脚本
└─ output/                        导入过程的中间产物（Markdown、图片、评审记录）
```

---

## 四、怎么跑起来

### 方式 A：Docker（推荐，一条命令起 7 个服务）

```bash
cp docker-compose.example.yml docker-compose.yml
cp .env.example .env
# 编辑 .env：至少填 OPENAI_API_KEY、VL_API_KEY、MCP_DASHSCOPE_API_KEY、MINERU_API_TOKEN，
#            以及 MONGO_ROOT_PASSWORD / MINIO_SECRET_KEY 这类口令（禁止沿用示例值）
docker compose up -d --build
```

| 服务 | 端口 | 入口 |
| --- | --- | --- |
| import-api | 8010 | http://127.0.0.1:8010/import.html |
| query-api | 8011 | http://127.0.0.1:8011/chat.html |
| Attu（Milvus 控制台） | 8000 | http://127.0.0.1:8000 |
| MinIO 控制台 | 9001 | http://127.0.0.1:9001 |
| Mongo | 27018 | `mongodb://admin:***@127.0.0.1:27018` |
| Milvus | 19530 | — |

端口这么分是因为 **8000 给了 Attu**，两个 API 只能往后排。

### 方式 B：本机跑两个 FastAPI

依赖服务仍要先用 compose 起（`etcd` / `minio` / `standalone` / `mongo`），然后在项目根目录：

```bash
uv sync                                  # 或 pip install -r requirements.lock.txt
python main.py --host 127.0.0.1 --port 8010
python -m app.query_process.api.query_service    # 8011
```

`start_all.ps1` 用来把这套流程串成一次点击，但**脚本里的 `$PROJECT` 是作者本机路径**（`D:\PythonProject\knowledge_base`），克隆到别的目录必须先改这一行，否则会去找一个不存在的 `.venv`。

---

## 五、环境变量

完整清单见 `.env.example`，几个必须自己填的：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_API_BASE` | 文本模型；默认指向百炼的 OpenAI 兼容地址 |
| `VL_API_KEY` / `VL_MODEL` | 图片摘要用的视觉模型（默认 `qwen-vl-max`；未配置时回退主 key） |
| `MINERU_API_TOKEN` | MinerU 解析 PDF（云端 API） |
| `MCP_DASHSCOPE_API_KEY` | 联网兜底的 MCP 检索（缺省时复用 `VL_API_KEY`） |
| `BGE_M3_PATH` / `BGE_RERANKER_LARGE` | 两个本地模型目录，不随仓库分发 |
| `MONGO_URL` / `MONGO_ROOT_PASSWORD` | 历史会话存储 |
| `WRITE_API_KEY` | 写接口的简单鉴权 |

模型默认跑 GPU（`BGE_DEVICE=cuda`、`BGE_FP16=true`）；只有 CPU 的机器要改成 `cpu` 并接受慢很多。

---

## 六、怎么验证它真的能跑

仓库里只有一个链路脚本，跑法：

```bash
python test/04-test_graph_flow.py
```

它会用 `stream_mode=["updates","values"]` 同时订阅两种流，逐节点打印「节点名 → 该节点的状态增量」，最后给出跑过的节点序列和终态。**能看出流程走通、每一步返回了什么**，这是它的用途。

脚本里的示例文件 `万用表RS-12的使用.pdf` 是作者本机的样本，仓库不附带 PDF；换成本地任意一份 PDF 的路径即可。

它**不校验检索质量**，也没有断言——见下节第 1 条。

---

## 七、已知边界

1. **没有 pytest 套件**。全仓库只有 `test/04-test_graph_flow.py` 一个链路脚本，没有断言、没有覆盖率；改代码后是否回归，靠人眼看节点输出。这是最需要补的一块。
2. **模型不随仓库分发**。BGE-M3、bge-reranker-large 加起来数 GB，只给了路径变量；缺模型时会在加载阶段明确报错，不会静默降级。
3. **默认要 GPU**。`BGE_DEVICE=cuda`，纯 CPU 环境需改配置，且一份 PDF 的向量化时间会明显变长。
4. **外部服务需自备**：MinerU Token、百炼 API Key、MCP 额度。缺任何一个，对应节点会失败并打印原因，不会返回编造结果。
5. **`torch` 线程数必须在 import 前锁定**。`main.py` 第一行导入 `app.core.torch_env`，就是为了抢在任何 `torch` 初始化之前设置 MKL/OpenMP 线程数；顺序一旦被打乱，BGE-M3 向量化会 Segmentation fault。这是设计约束，不是可以随手挪的 import。
6. **`start_all.ps1` 的路径是写死的**，见第四节方式 B 的说明。

---

## 八、许可

仓库未附许可证文件。代码可读、可参考；如需商用请先联系。
