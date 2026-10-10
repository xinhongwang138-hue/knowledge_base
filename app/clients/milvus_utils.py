"""
Milvus 客户端工具

对外提供：
    1. get_milvus_client()                  —— 连接单例，避免重复建连
    2. fetch_chunks_by_chunk_ids()          —— 按 chunk_id 批量回捞切片（供检索侧补全文本）
    3. create_hybrid_search_requests()      —— 构建稠密/稀疏双路ANN 检索请求
    4. hybrid_search()                      —— WeightedRanker 加权融合的混合检索

说明：连接失败统一返回 None / 空列表，由调用方决定降级策略，
     避免单个检索场景把整条链路打断。
"""
from typing import List, Tuple

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from app.clients.milvus_config import milvus_config
from app.core.logger import logger

# 全局 Milvus 客户端实例，实现单例复用
_milvus_client = None


def get_milvus_client():
    """
    Milvus 客户端单例获取方法

    实现客户端连接复用，避免重复创建连接消耗资源。
    :return: MilvusClient 实例，连接失败返回 None
    """
    global _milvus_client
    # 单例判断：已初始化则直接复用
    if _milvus_client is not None:
        return _milvus_client

    try:
        milvus_uri = milvus_config.milvus_url
        # 校验 Milvus 连接地址配置
        if not milvus_uri:
            logger.error("Milvus 客户端连接失败：缺少 MILVUS_URL 环境变量配置")
            return None
        _milvus_client = MilvusClient(uri=milvus_uri)
        logger.info(f"Milvus 客户端连接成功：{milvus_uri}")
        return _milvus_client
    except Exception as exc:
        logger.error(f"Milvus 客户端连接异常：{str(exc)}")
        return None


def _coerce_int64_ids(ids) -> Tuple[List[int], List]:
    """
    转换 chunk_id 为 Milvus 要求的 INT64 类型（主键字段 schema 为 INT64）

    过滤无效 ID，分离可转换/不可转换的 ID。
    :param ids: 待转换的 chunk_id 列表
    :return: (ok_ids, bad_ids)，ok_ids 为可转换的 int64 ID 列表，bad_ids 为无效 ID 列表
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except (TypeError, ValueError):
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
    client,
    collection_name: str,
    chunk_ids,
    *,
    output_fields=None,
    batch_size: int = 100,
):
    """
    通过 chunk_id 主键批量查询 Milvus 中的切片数据

    用于补全「仅拥有 chunk_id 无文本内容」场景的切片信息。
    优先使用 get 方法（主键直查，性能最优），失败则回退 query 过滤查询。
    :param client: MilvusClient 实例
    :param collection_name: 集合名称
    :param chunk_ids: 待查询的 chunk_id 列表
    :param output_fields: 需要返回的字段列表，默认返回核心切片字段
    :param batch_size: 分批查询大小，避免单次数据量过大，默认 100
    :return: List[dict]，Milvus 实体字典列表，查询失败返回空列表
    """
    # 前置校验：客户端/集合名无效直接返回空
    if client is None or not collection_name:
        return []

    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # 转换 ID 为 INT64 类型，分离有效/无效 ID
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"存在无法转换为 INT64 的 chunk_id，将跳过查询：{bad_ids}")
    if not ok_ids:
        return []

    results = []
    # 分批查询，避免单次 where 条件过长
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # 方式1：优先使用主键 get 方法查询（性能最优）
        if hasattr(client, "get"):
            try:
                got = client.get(
                    collection_name=collection_name,
                    ids=batch,
                    output_fields=output_fields,
                )
                if got:
                    results.extend(got)
                continue
            except Exception as exc:
                logger.warning(f"Milvus get 方法查询失败，将回退至 query 方法：{str(exc)}")

        # 方式2：get 不可用或失败，回退使用 filter 过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            queried = client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
            )
            if queried:
                results.extend(queried)
        except Exception as exc:
            logger.error(f"Milvus query 方法批量查询 chunk_id 失败：{str(exc)}")

    return results


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    """
    构建 Milvus 混合搜索请求对象

    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合。
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量（{特征索引: 权重} 字典）
    :param dense_params: 稠密向量搜索参数，默认余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认内积
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认 5
    :return: 搜索请求列表 [dense_req, sparse_req]
    """
    # 稠密向量默认用 COSINE，稀疏向量默认用 IP，均为 BGE-M3 的推荐搭配
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )
    return [dense_req, sparse_req]


def hybrid_search(
    client,
    collection_name,
    reqs,
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5,
    output_fields=None,
    search_params=None,
):
    """
    执行 Milvus 稠密+稀疏向量混合搜索

    基于 WeightedRanker 实现双向量搜索结果加权融合，提升检索准确性。
    :param client: MilvusClient 实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为 [dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，依次对应稠密/稀疏
    :param norm_score: 是否归一化评分后再融合，避免量级差异导致权重失效
    :param limit: 最终返回结果数量，默认 5
    :param output_fields: 需要返回的字段列表，默认返回 item_name
    :param search_params: 搜索参数，如 ef/topk 等
    :return: 混合搜索结果列表，搜索失败返回 None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏两路结果
        rerank = WeightedRanker(
            ranker_weights[0],
            ranker_weights[1],
            norm_score=norm_score,
        )

        if output_fields is None:
            output_fields = ["item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

        logger.info(f"Milvus 混合搜索完成，集合[{collection_name}]共检索到 {len(res)} 条结果")
        return res
    except Exception as exc:
        logger.error(f"Milvus 混合搜索执行失败，集合[{collection_name}]：{str(exc)}")
        return None


if __name__ == "__main__":
    cli = get_milvus_client()
    if cli is None:
        print("Milvus 连接失败，请检查 MILVUS_URL 与服务状态")
    else:
        print(f"Milvus 连接成功：{milvus_config.milvus_url}")
        print("现有集合：", cli.list_collections())
        print("INT64 转换自测：", _coerce_int64_ids([1, "2", "abc", None, 3.0]))