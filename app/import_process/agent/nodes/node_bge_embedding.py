"""
    主要目标：
         用 BGE-M3 模型把文本切片 chunks 转换为机器可计算的向量（稠密+稀疏双向量），
         为下游 node_import_milvus 入库 Milvus 做准备。这是 RAG 从「文本」走向「可检索」的关键桥梁。
    实现步骤：
         1. 输入校验：验证 chunks 有效性，核心数据缺失则终止当前节点
         2. 模型初始化：获取 BGE-M3 单例模型实例，避免重复加载
         3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
         4. 状态更新：将带向量的 chunks 更新回全局状态，供下游 Milvus 入库节点使用
"""
from typing import Any, List, Dict

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings, get_bge_m3_ef
from app.utils.task_utils import add_done_task, add_running_task

# --- 配置参数 (Configuration) ---
# 每批处理的切片数：批次太大会导致内存/显存压力骤增，太小则吞吐低。
# BGE-M3 是 568M 参数的大模型，CPU 上单批 5 条是稳定性与效率的平衡点；有显卡时可调大
EMBED_BATCH_SIZE = 5


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    """
    # 日志和任务队列处理
    add_running_task(state['task_id'], "node_bge_embedding")
    try:
        # 步骤1：输入数据校验，核心 chunks 无效则抛出异常
        texts_to_embed = step_1_validate_input(state)
        # 步骤2：初始化 BGE-M3 模型（单例模式，仅加载一次）
        bge_m3_ef = step_2_init_model()
        # 步骤3：批量生成双向量，为切片绑定向量字段
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
        # 步骤4: 输出数据处理
        state['chunks'] = output_data
        logger.info(f"向量化完成，共 {len(output_data)} 个切片已绑定向量")
        add_done_task(state['task_id'], "node_bge_embedding")
        return state
    except Exception as exc:
        logger.error(f"向量化节点执行失败：{str(exc)}")
        add_done_task(state['task_id'], "node_bge_embedding", status='failed')
        raise


@step_log("step_1_validate_input")
def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    步骤1：输入数据有效性校验
    核心作用：
        1. 从全局状态提取待向量化的 chunks 切片列表
        2. 严格校验 chunks 类型和非空性，无有效数据则终止向量化
    :param state: 图状态字典
    :return: 校验通过的切片列表
    :raise ValueError: chunks 为空或非列表时抛出
    """
    # 从状态中提取切片数据
    texts_to_embed = state.get("chunks")
    # 校验：必须是非空列表，否则无法进行向量化
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error("向量化输入校验失败：chunks 字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")

    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
    return texts_to_embed


@step_log("step_2_init_model")
def step_2_init_model():
    """
    步骤2：初始化 BGE-M3 模型实例（单例模式）
    核心作用：
        1. 调用单例函数 get_bge_m3_ef，确保模型全局仅加载一次
        2. 校验模型实例有效性，加载失败则抛出明确异常
    :return: 有效的 BGE-M3 模型实例（embedding function）
    :raise ValueError: 模型加载失败（路径错误/显存不足/依赖缺失）时抛出，并提示配置方向
    """
    try:
        # 获取单例模型实例，避免重复加载浪费资源
        ef = get_bge_m3_ef()
        # 校验模型实例是否有效
        if ef is None:
            raise ValueError("BGE-M3 模型实例为 None：pymilvus.model 模块未找到或模型加载失败")

        logger.info("BGE-M3 模型实例初始化成功（单例模式）")
        return ef
    except Exception as exc:
        # 包装异常信息，明确错误原因和排查方向
        error_msg = (
            f"BGE-M3 模型初始化失败：{exc}，"
            f"请检查模型路径（BGE_M3_PATH）/设备（BGE_DEVICE）/环境变量配置是否正确"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


@step_log("step_3_generate_embeddings")
def step_3_generate_embeddings(texts_to_embed, bge_m3_ef) -> List[Dict[str, Any]]:
    """
    步骤3：批量生成双向量（分批执行，每批独立异常处理）

    核心逻辑：
       1. 文本拼接: item_name + content 拼接，强化核心特征
       2. 批量调用: 传入拼接后的文本，生成批量双向量
       3. 向量绑定: 为每个切片复制原数据，新增 dense_vector 和 sparse_vector 字段
       4. 异常兜底: 每批次发生异常不影响全局处理

    :param texts_to_embed: 校验通过的文本切片列表，含 item_name / content 字段
    :param bge_m3_ef: 步骤2 初始化的 BGE-M3 模型实例（此处由 generate_embeddings 内部复用单例，
                      保留入参是为了显式表达「依赖已就绪」，避免后续误删模型初始化步骤）
    :return: 带向量字段的文本切片列表，异常批次保留原数据
    """
    # 初始化结果列表，存储带有向量的 chunk 数据
    output_data = []
    batch_size = EMBED_BATCH_SIZE
    total = len(texts_to_embed)

    # 循环进行批量处理（使用循环步长）
    for i in range(0, total, batch_size):
        # 截取当前批次的切片，最后一批自动适配剩余数量【每次获取 batch_size 个】
        batch_texts = texts_to_embed[i: i + batch_size]
        # 使用异常捕捉进行批量处理
        try:
            # 构建模型输入文本: item_name + content 拼接，强化核心特征
            # （几乎所有 Embedding 模型对前 128 个 token 的注意力最集中，越靠后的词对向量方向拉扯越弱，
            #  所以遵循「核心词前置」原则；另外 Embedding 模型对标点语言很敏感，
            #  喂中文就用全套中文标点，保持语境纯粹，向量质量最高）
            input_texts = []
            for chunk in batch_texts:
                item_name = chunk.get("item_name") or ""
                content = chunk.get("content") or ""
                # 有商品名则拼接（中文标点做分隔），无则直接使用内容
                text = f"商品：{item_name}，介绍：{content}" if item_name else content
                input_texts.append(text)

            # 调用封装函数生成批量向量
            # 返回格式：{"dense": [稠密向量列表], "sparse": [{特征索引: 权重}]}
            docs_embeddings = generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.error("向量化结果为空：请检查输入文本是否为空")
                output_data.extend(batch_texts)
                continue
            # 为当前批次每个切片绑定对应向量，复制原数据避免污染上游 state
            for j, doc in enumerate(batch_texts):
                item = dict(doc)
                item["dense_vector"] = docs_embeddings["dense"][j]  # 绑定稠密向量
                item["sparse_vector"] = docs_embeddings["sparse"][j]  # 绑定稀疏向量
                output_data.append(item)

        except Exception as exc:
            # 捕获异常，记录错误信息并跳过当前批次
            logger.error(f"向量化批次处理异常：{exc}")
            # 异常批次保留原切片数据，保证数据完整性，后续可人工排查
            output_data.extend(batch_texts)
            # 跳过当前批次，继续处理下一批次
            continue

    # 统计本次向量化结果，失败的批次没有向量字段，便于下游与人工排查
    ok_count = sum(1 for c in output_data if "dense_vector" in c)
    fail_count = len(output_data) - ok_count
    if fail_count:
        logger.warning(f"向量化完成：成功 {ok_count} 个切片，失败 {fail_count} 个（失败批次已保留原文）")
    return output_data


# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个 LangGraph 流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
if __name__ == '__main__':
    from dotenv import load_dotenv

    from app.core.logger import PROJECT_ROOT

    # 加载环境变量：定位项目根目录下的 .env，读取模型路径/设备等配置
    # 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
    load_dotenv(PROJECT_ROOT / ".env")

    # 构造模拟测试状态：模拟上游节点输出的 chunks 数据，贴合真实业务场景
    test_state: ImportGraphState = {
        "task_id": "test_task_embedding_001",  # 测试任务ID
        # 模拟带 item_name 的文本切片（上游主体识别节点产出）
        "chunks": [
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf",
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf",
            }
        ]
    }

    # 执行本地测试
    logger.info("=== BGE-M3 向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info("=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | "
                f"稀疏向量生成{'' if has_sparse else '未'}成功"
            )
    except Exception as exc:
        logger.error(f"=== 向量化节点本地测试失败 === 错误原因：{str(exc)}")
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查 BGE-M3 模型路径、显存是否充足、环境变量配置是否正确")
