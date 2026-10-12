"""
BGE-M3 向量生成工具

BGE-M3 是多语言/多粒度嵌入模型，单次编码同时输出两种向量：
    - dense（稠密向量）：定长 1024 维，捕捉语义，适合相似度计算
    - sparse（稀疏向量）：维度极高但只有少量非零，捕捉字面关键词，适合精准召回

两路配合即「混合检索（Hybrid Search）」，兼顾语义理解与字面匹配。

对外提供：
    1. get_bge_m3_ef()      —— 模型单例，避免重复加载（加载一次约需数秒~数十秒）
    2. generate_embeddings()—— 批量文本 -> {dense: [...], sparse: [{idx: weight}, ...]}
"""
# 【重要】必须早于任何 torch import 生效，否则 BGE-M3 前向计算会 Segmentation fault。
# 具体原理与排查过程见 app/core/torch_env.py 的模块文档。
from app.core.torch_env import TORCH_ENV_INITIALIZED  # noqa: F401  导入即生效

import threading
from typing import TYPE_CHECKING

from app.core.logger import logger
from app.lm.embedding_config import embedding_config

# 仅供静态分析（PyCharm）使用的类型提示导入：
# TYPE_CHECKING 恒为 False，运行时的真实导入仍在函数内延迟发生，
# 这样既让 IDE 能解析符号消除误报，又不会让纯 LLM 场景白白加载 torch。
if TYPE_CHECKING:
    # noqa 标注：这是给 IDE 看的类型提示，运行时不会执行
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction  # noqa: F401

# 模型单例对象，避免重复初始化（懒加载：只有真正调用时才加载模型）
_bge_m3_ef = None

# embedding 前向互斥锁：与 reranker_utils.release_embedding_model() 协作，
# 保证「加载/编码」与「offload 释放」互斥，避免模型被搬到 meta 设备后
# 仍被并发的前向调用触发 "Cannot copy out of meta tensor" 报错。
# 释放方以非阻塞方式尝试获取，拿不到锁则跳过释放（宁可不释放也不打断检索）。
_EMBED_USE_LOCK = threading.Lock()


def get_bge_m3_ef():
    """
    获取 BGE-M3 模型单例对象，自动加载环境变量配置

    :return: 初始化完成的 BGEM3EmbeddingFunction 实例
    :raise: 模型初始化失败时向上抛出异常，由调用方降级处理
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 本地有模型就用本地路径，没有则按仓库标识 "BAAI/bge-m3" 自动下载
    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    logger.info(
        f"开始初始化 BGE-M3 模型：model_name={model_name}, device={device}, use_fp16={use_fp16}"
    )

    try:
        # 延迟导入：FlagEmbedding 依赖链较重（会连带导入 torch/transformers），
        # 放在函数内可避免仅使用 LLM 功能的场景白白付出导入开销
        #
        # 下面这行TYPE_CHECKING 只在「静态分析时」执行、运行时为 False，
        # 因此真实行为仍是函数内延迟导入（不引入torch），
        # 但 PyCharm 能据此解析符号，不再报「找不到模块」和「找不到引用」。
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        # 初始化模型，开启原生 L2 归一化（适配 Milvus IP/COSINE 内积检索）
        # 归一化后向量模长为 1，IP 与余弦相似度等价，检索更稳且两路分数便于融合
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        logger.info("BGE-M3 模型初始化成功，已开启原生 L2 归一化")
        return _bge_m3_ef
    except Exception as exc:
        logger.error(f"BGE-M3 模型初始化失败：{str(exc)}")
        raise


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生 L2 归一化）

    :param texts: 要生成嵌入的文本列表，单条文本也需封装为列表
    :return: 字典格式的向量结果：
             {"dense": [向量列表, ...], "sparse": [{特征索引: 权重}, ...]}
             两路结果与输入文本严格一一对应
    :raise: 入参不合法抛 ValueError；编码过程异常向上抛出
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts 必须为非空列表")
        raise ValueError("参数 texts 必须是包含文本的非空列表")

    logger.info(f"开始为 {len(texts)} 条文本生成混合向量嵌入")
    try:
        # 持锁完成「获取模型 + 前向编码」：锁内保证模型不被 reranker 路径
        # 的 release_embedding_model() 并发 offload 到 meta 设备。
        with _EMBED_USE_LOCK:
            model = get_bge_m3_ef()
            # 模型编码，返回 dense（稠密向量）+ sparse（CSR 格式稀疏向量）
            embeddings = model.encode_documents(texts)

        # 把 CSR 稀疏矩阵按「每条文本一行」拆成 {特征索引: 权重} 字典
        #   indices：非零元素的列号（特征 ID）
        #   data：该列号对应的权重值
        #   indptr：每一行在 indices/data 中的起止位置指针
        # 数据示例：indices=[3,8,20,1,9]、data=[0.7,0.2,0.1,0.6,0.4]、indptr=[0,3,5]
        #   第 0 条文本取 0:3 -> indices=[3,8,20], data=[0.7,0.2,0.1]
        #   第 1 条文本取 3:5 -> indices=[1,9], data=[0.6,0.4]
        processed_sparse = []
        for i in range(len(texts)):
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()  # .tolist() 把 np.int64 转成 Python int，满足字典 key 可哈希要求
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()  # .tolist() 把 np.float32 转成 Python float，可 JSON 序列化
            processed_sparse.append(dict(zip(sparse_indices, sparse_data)))

        # 稠密向量同样转列表，解决 numpy 数组不可序列化问题
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": processed_sparse,
        }
        logger.info(f"{len(texts)} 条文本向量生成完成（dense 维度={len(result['dense'][0])}）")
        return result
    except Exception as exc:
        logger.error(f"文本向量生成失败：{str(exc)}")
        raise


"""
核心设计亮点：
1. normalize_embeddings=True 的价值：
   - 检索更稳定：不同文本长短、词频差异不会把分数拉偏；
   - IP 近似 cosine：向量都归一化后内积与余弦等价，Milvus 用 IP/COSINE 检索都合适；
   - dense/sparse 统一标尺：混合检索时两路分数便于融合，不会一边压死另一边；
   - 减少异常高分：防止「模长大」的向量仅靠长度拿高分。
2. NumPy 类型转原生类型：sparse 的 index/value 加 .tolist()，避免 np.int64 做 key 报错、
   np.float32 无法 JSON 序列化的问题，可直接入库或走接口返回。
3. 单例模式：模型仅初始化一次，避免重复加载耗时耗资源。
4. 格式对齐业务：返回 dense 嵌套列表 + sparse 字典列表，
   与调用方 result['dense'][0] / result['sparse'][0] 的取值逻辑一致。
5. 延迟导入：FlagEmbedding 只在真正需要向量时才加载，纯 LLM 场景不承担这部分导入开销。
6. 不吞异常：编码失败直接抛出，由调用方做重试或降级，便于问题定位。
"""


if __name__ == "__main__":
    demo = ["华为Mate60 Pro手机", "苏泊尔电磁炉"]
    res = generate_embeddings(demo)
    print(f"dense 维度: {len(res['dense'][0])}")
    print(f"sparse 非零特征数: {[len(s) for s in res['sparse']]}")