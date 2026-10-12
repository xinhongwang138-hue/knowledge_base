"""
稀疏向量归一化工具

背景：BGE-M3 输出的稀疏向量权重不保证模长为 1，而 Milvus 对稀疏字段
     建索引时要求用 IP（内积）度量。若权重忽大忽小，融合打分时
     稀疏通道会被个别高权重词「压过」稠密通道。

策略：对稀疏向量权重整体做 L2 归一化，使稀疏通道与稠密通道的分数量级可比，
     让 WeightedRanker 的权重设置真正生效。
"""
import math


def normalize_sparse_vector(sparse_vector):
    """
    对单条稀疏向量做 L2 归一化

    :param sparse_vector: 稀疏向量，形如 {特征索引: 权重} 的字典
    :return: 归一化后的同结构字典；输入为空或全零时原样返回
    """
    if not sparse_vector:
        return sparse_vector

    # 计算所有权重的平方和（即 L2 范数的平方）
    sum_of_squares = sum(float(weight) ** 2 for weight in sparse_vector.values())
    if sum_of_squares <= 0:
        # 全零向量无法归一化，原样返回
        return sparse_vector

    norm = math.sqrt(sum_of_squares)
    # 除以范数，让模长归一到 1
    return {idx: float(weight) / norm for idx, weight in sparse_vector.items()}


if __name__ == "__main__":
    demo = {3: 0.7, 8: 0.2, 20: 0.1, 1: 0.6, 9: 0.4}
    normed = normalize_sparse_vector(demo)
    norm = math.sqrt(sum(v ** 2 for v in normed.values()))
    print(f"原始: {demo}")
    print(f"归一: {normed}")
    print(f"归一后模长: {norm:.6f}（应≈1.0）")
    print(f"空输入: {normalize_sparse_vector({})}")