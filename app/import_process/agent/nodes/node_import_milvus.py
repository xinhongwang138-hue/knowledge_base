"""
    主要目标：
         把上游 node_bge_embedding 产出的「带向量切片」持久化写入 Milvus，
         构建可供即时查询的索引。这是整个知识库导入流程的数据落地点。
    实现步骤：
         1. 检查数据 chunks 是否存在
         2. 前置准备工作：准备 Milvus 的集合和字段（不存在则创建）
         3. 删除旧数据（按 item_name 清理，保证幂等）
         4. 插入 chunks 数据并回填 Milvus 主键
"""
import hashlib
from typing import Any, Dict, List

from pymilvus import DataType

from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import get_milvus_client
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 从配置读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    """
    # 准备日志和任务列表
    add_running_task(state['task_id'], "node_import_milvus")
    try:
        # 1. 检查数据 chunks 是否存在
        chunks = step_1_check_input(state)
        # 2. 前置准备工作：创建/确认 Milvus 集合和字段
        milvus_client = get_milvus_client()
        if milvus_client is None:
            raise RuntimeError("Milvus 客户端不可用，请检查 MILVUS_URL 配置与 Milvus 服务状态")
        step_2_prepare_collection(milvus_client)
        # 3. 删除旧数据（按本批 chunks 涉及的全部 item_name 清理，保证幂等）
        step_3_delete_old_data(milvus_client, state.get('item_name'), chunks)
        # 4. 插入 chunks 的数据，并把生成的主键回填到 chunk
        with_id_chunks = step_4_insert_collections(milvus_client, chunks)
        state['chunks'] = with_id_chunks
        add_done_task(state['task_id'], 'node_import_milvus')
        return state
    except Exception as exc:
        logger.error(f"导入 Milvus 节点执行失败：{str(exc)}")
        add_done_task(state['task_id'], 'node_import_milvus', status='failed')
        raise


@step_log("step_1_check_input")
def step_1_check_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    步骤1：检查输入数据 chunks 是否存在

    :param state: 图状态字典
    :return: 校验通过的切片列表
    :raise ValueError: chunks 不存在时抛出
    """
    chunks = state.get('chunks')
    if not chunks:
        logger.error("node_import_milvus: chunks 数据不存在")
        raise ValueError("node_import_milvus: chunks 数据不存在")

    # 上游契约校验：没有向量的切片写进 Milvus 毫无意义，提前拦截比让 Milvus 报错清晰得多
    has_vector = sum(1 for c in chunks if c.get("dense_vector") and c.get("sparse_vector"))
    if has_vector == 0:
        raise ValueError(
            "所有切片都缺少向量字段（dense_vector / sparse_vector），"
            "请确认上游 node_bge_embedding 已执行成功"
        )
    if has_vector < len(chunks):
        logger.warning(
            f"部分切片缺少向量：{len(chunks) - has_vector}/{len(chunks)} 条，"
            f"这些切片将被跳过（通常来自 node_bge_embedding 的失败批次）"
        )

    logger.info(f"导入输入校验通过，待写入切片数量：{len(chunks)}（含向量 {has_vector} 条）")
    return chunks


@step_log("step_2_prepare_collection")
def step_2_prepare_collection(milvus_client):
    """
    步骤2：准备和创建 chunks 对应的集合

    集合不存在时按以下 schema 创建（字段与索引严格对齐检索侧约定）：
        chunk_id      INT64                主键，自增
        content       VARCHAR(65535)       切片正文
        title         VARCHAR(65535)       切片标题
        parent_title  VARCHAR(65535)       父级标题
        part          INT8                 子块序号
        file_title    VARCHAR(65535)       来源文件名
        item_name     VARCHAR(65535)       文档主体名（幂等清理依据）
        dense_vector  FLOAT_VECTOR(1024)   稠密向量，COSINE + HNSW 索引
        sparse_vector SPARSE_FLOAT_VECTOR  稀疏向量，IP + 倒排索引

    :param milvus_client: MilvusClient 实例
    :return: 同一个 milvus_client，方便链式调用
    """
    if not milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        # 创建集合对应的列的信息（主键自增长 + 动态字段，
        # 动态字段让后续新增元数据字段不必改表结构）
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # P2-2：doc_id 作为文件级幂等键（sha1(item_name) 派生），
        # 同一文件重新导入时 doc_id 恒定，可按 doc_id 精准清理旧数据，
        # 避免仅按 item_name 清理时跨主体残留。
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 配置索引，保证混合检索（Hybrid Search）可用
        index_params = milvus_client.prepare_index_params()

        #稠密索引：HNSW。M 越大召回越高，但内存占用与插入耗时也越高。
        #   1万数据量级  M=16, efConstruction=200
        #   5万数据量级  M=32, efConstruction=300（笔记中切片集合采用此档）
        #   10万数据量级 M=64, efConstruction=400
        # efConstruction: 建索引时考虑的候选节点数量，越高图质量越好但构建越慢
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 300},
        )
        # 稀疏索引：倒排索引 + DAAT_MAXSCORE，只计算可能得高分的向量，跳过大量 0
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        milvus_client.create_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            schema=schema,# 字段
            index_params=index_params,  # 索引
        )
        logger.info(f"Milvus 集合 [{CHUNKS_COLLECTION_NAME}] 不存在，已创建")
    else:
        logger.info(f"Milvus 集合 [{CHUNKS_COLLECTION_NAME}] 已存在，直接复用")

    # 加载集合，后续读写都基于加载态
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
    return milvus_client


def _collection_has_field(milvus_client, field_name: str) -> bool:
    """检测集合 schema 是否包含指定字段（兼容 describe 返回 dict / 对象）"""
    try:
        desc = milvus_client.describe_collection(collection_name=CHUNKS_COLLECTION_NAME)
        fields = desc.get("fields") if isinstance(desc, dict) else getattr(desc, "fields", [])
        return any(
            (f.get("name") if isinstance(f, dict) else getattr(f, "name", None)) == field_name
            for f in fields
        )
    except Exception as exc:
        logger.warning(f"检测集合字段失败（{field_name}）：{exc}")
        return False


def _derive_doc_id(item_name) -> str:
    """从 item_name 派生稳定 doc_id（同一文档主体重复导入时 doc_id 恒定）"""
    return hashlib.sha1((item_name or "").encode("utf-8")).hexdigest()


@step_log("step_3_delete_old_data")
def step_3_delete_old_data(milvus_client, item_name, chunks=None):
    """
    步骤3：删除已存在的旧切片（P2-2：优先按 doc_id，无则按 item_name 兜底）

    幂等说明：不能只删 state 里的那一个 item_name。
    同一批 chunks 中每条切片的 item_name 可能不同（例如逐条识别或历史数据残留），
    只删 state.item_name 会让其它 item_name 的旧切片在重复导入时不断累积。
    因此：
      - 新集合 schema 含 doc_id 字段时，按文件级 doc_id 清理（同一文件 doc_id 恒定，
        跨 item_name 的历史残留也能一次清干净）；
      - 旧集合无该字段时，维持以 state.item_name 为主、chunks 内全部 item_name 一并清理。

    :param milvus_client: MilvusClient 实例
    :param item_name: 文档主体名（来自 state）
    :param chunks: 本次待写入的切片列表，用于收集其中出现过的全部 item_name
    """
    # 优先路径：按 doc_id 文件级清理（新集合）
    has_doc_id_field = _collection_has_field(milvus_client, "doc_id")
    if has_doc_id_field:
        doc_ids = list(
            dict.fromkeys(
                c.get("doc_id") or _derive_doc_id(c.get("item_name"))
                for c in (chunks or [])
            )
        )
        doc_ids = [d for d in doc_ids if d]
        if doc_ids:
            quoted = [f"'{escape_milvus_string(d)}'" for d in doc_ids]
            milvus_client.delete(
                collection_name=CHUNKS_COLLECTION_NAME,
                filter=f"doc_id in [{', '.join(quoted)}]",
            )
            # load_collection() 会触发 Milvus 重新加载集合数据、刷新索引、清理已标记删除的数据，
            # 确保删除操作真正生效，避免新旧数据混杂导致检索错误
            milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
            logger.info(f"已按 doc_id 清理旧切片数据（{len(quoted)} 个 doc_id）")
            return

    # 兜底路径：按 item_name 清理（旧集合 / chunks 无 doc_id 场景）
    # 汇总本次会写入的全部 item_name（去重且丢弃空值）
    targets = []
    for name in [item_name] + [c.get("item_name") for c in (chunks or [])]:
        if name and name not in targets:
            targets.append(name)

    if not targets:
        logger.warning("item_name 为空，跳过旧数据清理（本次将直接追加写入）")
        return

    # item_name 来自 LLM/文件名，可能含引号等特殊字符，必须转义否则表达式会被截断
    quoted = [f"'{escape_milvus_string(name)}'" for name in targets]
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f"item_name in [{', '.join(quoted)}]",
    )
    # load_collection() 会触发 Milvus 重新加载集合数据、刷新索引、清理已标记删除的数据，
    # 确保删除操作真正生效，避免新旧数据混杂导致检索错误
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
    logger.info(f"已清理 item_name 属于 {targets} 的旧切片数据")


@step_log("step_4_insert_collections")
def step_4_insert_collections(milvus_client, chunks) -> List[Dict[str, Any]]:
    """
    步骤4：插入集合数据，并把 Milvus 生成的主键回填到 chunk

    :param milvus_client: MilvusClient 实例
    :param chunks: 带向量的切片列表
    :return: 回填 chunk_id 后的切片列表（就地修改并返回同一对象）
    :raise ValueError: 切片缺少必填字段时抛出（Milvus 的报错难懂，这里提前拦截并给出明确提示）
    """
    # 集合 schema 中的必填字段（chunk_id 是自增主键，插入时反而不需要）
    required_fields = ("content", "title", "parent_title", "file_title", "item_name",
                       "dense_vector", "sparse_vector")
    # 集合 schema 中 part 为 INT8，缺失时给0 占位（序号本身不参与检索，容忍缺失）
    int8_fields = ("part",)
    # P2-2：新集合含 doc_id 字段时补写文件级幂等键（chunk 未携带则按 item_name 派生）
    has_doc_id_field = _collection_has_field(milvus_client, "doc_id")

    rows = []
    missing_report = set()
    for chunk in chunks:
        # 主键 chunk_id 由 Milvus 自增生成，插入前必须剔除临时/旧值，否则会被判为显式主键而报错
        row = {k: v for k, v in chunk.items() if k != "chunk_id"}

        # 补齐缺失的必填字段：VARCHAR 类用空串占位，避免 Milvus 直接抛
        # "Insert missed an field xxx" 这类难以定位的异常
        for field in required_fields:
            if field not in row or row[field] is None:
                if field not in missing_report:
                    logger.warning(f"切片缺少字段【{field}】，已用空值占位（chunk_index={len(rows)}）")
                    missing_report.add(field)
                if field in ("dense_vector", "sparse_vector"):
                    # 向量字段缺失无法用空值糊弄，单独提示并跳过该条
                    logger.error(f"切片缺少向量字段【{field}】，无法入库，已跳过该条")
                    row = None
                    break
                row[field] = {} if field == "sparse_vector" else ""
        if row is None:
            continue

        for field in int8_fields:
            if field not in row or row[field] is None:
                row[field] = 0

        # P2-2：补写 doc_id（仅当集合 schema 含该字段；旧集合不写，避免污染动态字段）
        if has_doc_id_field:
            row["doc_id"] = chunk.get("doc_id") or _derive_doc_id(row.get("item_name"))

        rows.append(row)

    if not rows:
        raise ValueError("所有切片都缺少向量字段（dense_vector / sparse_vector），无法写入 Milvus")

    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=rows)
    # 成功插入了几条
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据（提交 {len(rows)} 条）")

    # 获取回显的 ids 并回填
    ids = insert_result.get("ids", [])
    if ids and len(ids) == len(rows):
        for index, row in enumerate(rows):
            row["chunk_id"] = ids[index]
        logger.info(f"已回填 {len(ids)} 个 chunk_id 到切片数据")
    else:
        # 主键没回来不影响数据已写入的事实，只提示下游 chunk_id 暂缺
        logger.warning(f"未获取到完整的 chunk_id 回显（ids={len(ids)}, 数据={len(rows)}），跳过回填")

    return rows


# ==========================================
# 本地单元测试入口
# 功能：验证 Milvus 导入节点的完整流程（连接、建集合、清理旧数据、插入新数据）
# 适用场景：本地开发、调试、连通性验证
# ==========================================
if __name__ == '__main__':
    import os

    from dotenv import load_dotenv

    from app.core.logger import PROJECT_ROOT

    # 加载环境变量（用绝对路径定位项目根目录的 .env，避免依赖运行时工作目录）
    load_dotenv(PROJECT_ROOT / ".env")

    # 构造测试数据（dense/sparse 为模拟向量，用于验证入库链路而非向量质量）
    dim = 1024
    test_state: ImportGraphState = {
        "task_id": "test_milvus_task",
        "item_name": "测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8},  # 模拟 Sparse Vector
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                # 故意用不同的 item_name，验证同一次导入中不同主体不会被清理掉
                "item_name": "测试项目_Milvus2",
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")
    except Exception as exc:
        print(f"❌ 测试失败: {exc}")