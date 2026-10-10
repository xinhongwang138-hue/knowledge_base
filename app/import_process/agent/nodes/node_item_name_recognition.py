"""
    主要目标：
         1. 用文本大模型识别当前 chunks 对应的 item_name！用于区分不同的文档
         2. 使用BGE-M3 嵌入式模型，将 item_name 生成向量存储到向量数据库
         3. 修改 state['chunks'] -> chunk {title parent_title part file_title content item_name}
    实现步骤：
         1. 校验和取值（file_title, chunks）
         2. 构建上下文环境  chunks -> top 5 -> 拼接成 context 文本
         3. 调用模型，拼接提示词，识别 chunks 对应 item_name
         4. 修改 state chunks -> item_name
         5. item_name 生成向量（稠密/稀疏）
         6. 存储向量到向量数据库 kb_item_names（id / file_title / item_name / 稠密 和 稀疏）
"""
import os
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.clients.milvus_config import milvus_config
from app.clients.milvus_utils import get_milvus_client
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.llm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task, add_done_task

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前 5 个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认 2500
CONTEXT_TOTAL_MAX_CHARS = 2500
# LLM 调用失败时的重试次数
LLM_MAX_RETRY = 2


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    """
    # 日志和任务处理
    add_running_task(state['task_id'], 'node_item_name_recognition')
    try:
        # 1. 校验和取值（file_title, chunks）
        # 获取前置的材料！file_title = 为了兜底，没有识别到 item_name
        chunks, file_title = step_1_get_chunks_and_file_title(state)
        # 2. 构建上下文环境  chunks -> top 5 -> 拼接成 context 文本
        context = step_2_build_context(chunks)
        # 3. 调用模型，拼接提示词，识别 chunks 对应 item_name
        item_name = step_3_call_llm(context, file_title)
        # 4. 修改 state chunks -> item_name
        step_4_update_chunks_and_state(state, item_name, chunks)
        # 5. item_name 生成向量（稠密/稀疏）
        dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
        # 6. 将向量存储到向量数据库 kb_item_names
        step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

        add_done_task(state['task_id'], 'node_item_name_recognition')
        return state
    except Exception as exc:
        logger.error(f"主体识别节点执行失败：{str(exc)}")
        add_done_task(state['task_id'], 'node_item_name_recognition', status='failed')
        raise


@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state) -> Tuple[list, str]:
    """
    步骤1：校验并提取 chunks 与 file_title

    file_title 用于兜底：LLM 没识别出主体名时，用文件名顶上。
    :param state: 图状态字典
    :return: (chunks, file_title)
    :raise ValueError: chunks 为空时无法继续
    """
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    if not chunks:
        raise ValueError("chunks 没有值，无法继续进行，抛出异常处理！")

    if not file_title:
        # file_title 没有值！从 md_path 中获取文件名即可（字符串处理更方便）
        md_path = state.get('md_path')
        if md_path:
            file_title = os.path.splitext(os.path.basename(md_path))[0]
        if not file_title:
            # 两条路都拿不到就用任务 id 兜底，保证下游始终有主体名可用
            file_title = state.get('task_id') or "unknown_document"
            logger.warning("file_title 与 md_path 均为空，已用 task_id 兜底作为 file_title")
        state['file_title'] = file_title
        logger.info(f"state 中 file_title 为空，已回退推导：{file_title}")

    return chunks, file_title


@step_log("step_2_build_context")
def step_2_build_context(chunks) -> str:
    """
    步骤2：构建提示词上下文环境

    按 chunks 切面的 content 进行拼接，截取限制：
        1. 最多取前 top K 个（5）
        2. 总字符数不超过 CONTEXT_TOTAL_MAX_CHARS（2500）
    拼接格式：切片：{i}，标题:{title},内容：{content}
    :param chunks: 上游切片列表
    :return: 拼接好的 context 文本
    """
    parts = []  # 存储处理后的切片文本
    total_chars = 0  # 记录已加入列表的字符数

    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunk_title = chunk.get('title', '')
        chunk_content = chunk.get('content', '')
        # 单切片内容先截断，避免一个超长切片直接吃掉整个上下文预算
        chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content}"
        parts.append(data)
        total_chars += len(data)
        # 达到总字符上限即停止拼接
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数：{total_chars}，停止拼接！")
            break

    context = "\n\n".join(parts)
    # 最终再兜底截断一次，防止最后一轮拼接超出上限
    final_context = context[:CONTEXT_TOTAL_MAX_CHARS]
    logger.info(f"构建上下文完成，共 {len(parts)} 个切片，{len(final_context)} 字符")
    return final_context


@step_log("step_3_call_llm")
def step_3_call_llm(context, file_title) -> str:
    """
    步骤3：调用大模型识别 item_name（含重试与兜底）

    :param context: 步骤2 拼好的上下文
    :param file_title: 文件名，用于兜底
    :return: 识别出的主体名称；识别失败返回 file_title
    """
    try:
        # 1. 构建提示词
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")
        messages = [
            HumanMessage(content=human_prompt),
            SystemMessage(content=system_prompt),
        ]
        # 2. 获取模型对象
        llm = get_llm_client(json_mode=False)
        # 3. 组装调用链（StrOutputParser 把 AIMessage 转成纯字符串）
        chain = llm | StrOutputParser()
        item_name = chain.invoke(messages)
        item_name = (item_name or "").strip()
        # 4. 判断：模型没给出有效名称时用 file_title 兜底
        if not item_name:
            logger.warning("大模型未返回有效的 item_name，使用 file_title 兜底")
            item_name = file_title
        else:
            # 去掉模型可能带上的引号/书名号等噪声符号
            item_name = item_name.strip('"\'""《》「」 ')
            logger.info(f"大模型识别 item_name 成功：{item_name}")
        return item_name
    except Exception as exc:
        # LLM 不稳（超时/限流/额度）不应中断整条导入链路，统一降级到 file_title
        logger.error(f"调用大模型识别 item_name 失败，使用 file_title 兜底：{str(exc)}")
        return file_title


@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    步骤4：把识别结果回填到 state 与每个 chunk

    :param state: 图状态字典
    :param item_name: 识别出的主体名称
    :param chunks: 切片列表，就地写入 item_name 字段
    """
    state['item_name'] = item_name

    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"完成了 chunks 和 state['item_name'] 的赋值和修改：{item_name}")


@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    """
    步骤5：根据 item_name 生成 BGE-M3 混合向量

    generate_embeddings 已封装模型调用：
        入参：文本 list["item_name"]
        返回：{"dense": [稠密向量], "sparse": [{idx: weight}]}
    稀疏向量额外做一次 L2 归一化，保证与稠密向量分数量级可比。
    :param item_name: 主体名称
    :return: (dense_vector, sparse_vector)
    """
    result = generate_embeddings([item_name])
    dense_vector, sparse_vector = result['dense'][0], result['sparse'][0]
    # 稀疏向量权重归一化，提升混合检索融合效果
    sparse_vector = normalize_sparse_vector(sparse_vector)
    logger.info(
        f"item_name 向量生成完成：dense维度={len(dense_vector)}, "
        f"sparse非零特征数={len(sparse_vector)}"
    )
    return dense_vector, sparse_vector


@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    步骤6：将 item_name 与向量写入 Milvus（保证幂等）

    幂等策略：先按 item_name 删除旧记录，再插入最新记录，
             避免同一主体重复入库导致后续检索出现重复项。
    :param file_title: 文件标题
    :param item_name: 主体名称
    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    """
    # 1. 获取 milvus 的客户端
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus 客户端不可用，无法保存 item_name 向量")

    collection_name = milvus_config.item_name_collection

    # 2. 判断集合（表）是否存在，不存在则创建集合（表）
    if not milvus_client.has_collection(collection_name=collection_name):
        # 3.1. 创建集合对应的列的信息（主键自增长 + 动态字段）
        schema = milvus_client.create_schema(
            auto_id=True,           # 主键自增长
            enable_dynamic_field=True,  # 动态字段，便于后续加字段不用改表结构
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # BGE-M3 稠密向量固定 1024 维
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3.2 配置索引
        index_params = milvus_client.prepare_index_params()
        # 稠密索引：HNSW，M 越大召回越高但内存与插入耗时也越高
        # 1万数据量级推荐 M=16, efConstruction=200
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        # 稀疏索引：倒排索引 + DAAT_MAXSCORE 只算可能得高分的向量，跳过大量 0
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,          # 字段
            index_params=index_params,  # 索引
        )
        logger.info(f"Milvus 集合 [{collection_name}] 不存在，已创建")

    # 加载集合后再读写
    milvus_client.load_collection(collection_name=collection_name)
    # 3. 先删除之前存在的 item_name（幂等：同主体只保留最新一条）
    # item_name 来自 LLM，需转义引号等特殊字符，否则表达式会被截断
    safe_item_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=collection_name,
        filter=f"item_name == '{safe_item_name}'",
    )
    # 4. 向集合插入最新的 item_name 数据和对应的向量
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }
    milvus_client.insert(collection_name=collection_name, data=[item])
    logger.info(f"保存了 item_name：{item_name} 的数据到向量数据库 [{collection_name}] 中！")


# ===================== 本地测试方法（直接运行调试，无需启动 LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法

    功能：模拟 LangGraph 流程输入，独立测试 node_item_name_recognition 节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个 LangGraph 流程
    测试前准备：
        1. 确保 .env 中 MILVUS_URL / ITEM_NAME_COLLECTION 等配置完成
        2. 确保大模型、Milvus、BGE-M3 模型均可正常访问
        3. 确保 prompt 模板（item_name_recognition / product_recognition_system）已存在
    使用方法：
        直接运行本文件：python -m app.import_process.agent.nodes.node_item_name_recognition
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的 ImportGraphState 状态（模拟上游节点产出数据）
        mock_state: ImportGraphState = {
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含 title/content 字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        }

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证 Milvus 存储
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION") or milvus_config.item_name_collection
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name == "{safe_name}"',
                output_fields=["file_title", "item_name"],
            )
            logger.info(f"Milvus 中检索到的数据：{res}")
    except Exception as exc:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(exc)}")


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    test_node_item_name_recognition()