"""
BGE Reranker 模型加载与打分工具

「模型本体」与「封装类」的分工：
    - BAAI/bge-reranker-large —— 相当于一台高性能发动机（权重 + 配置），提供重排序能力
    - FlagReranker            —— 相当于发动机的专用启动装置（封装了加载、设备适配、
                                 精度设置、推理调用），无需关心底层细节

【本机为什么不用 FlagReranker】
实测在本机（torch 2.10.0+cpu）用 FlagReranker 加载会在**权重加载阶段**直接段错误
（Windows 退出码 139，无 traceback）：
    - 该模型目录同时存在 model.safetensors 与 pytorch_model.bin
    - 走 safetensors 加载路径 -> 段错误
    - 走 pytorch_model.bin 加载路径 -> 完全正常（已实测 logits 形状 [2,1]）
结论是 torch 2.10 与该 safetensors 加载路径不兼容。因此这里直接用
transformers 的 AutoModelForSequenceClassification + use_safetensors=False，
功能等价（都是加载同一个 XLMRobertaForSequenceClassification 权重），
只是少了 FlagReranker 的那层薄封装。

为什么 Reranker 不能用 Embedding 替代：
    Embedding 把「问题」和「文档」分别编码成向量再比对（双塔），
    快但精度有限 —— 两句话语义相近、字面不同时容易漏；
    Reranker 把「问题 + 文档」拼成一句话一起进模型精读（交叉编码），
    能捕捉细微差异（如「苹果手机」vs「苹果公司」），代价是慢，
    所以只用于初筛后的小批量精排（Top 20~50），不做全库检索。
"""
import gc
import os
import threading

import torch

# 必须早于 torch 初始化：限制 MKL 线程，否则前向计算可能段错误
from app.core.torch_env import TORCH_ENV_INITIALIZED  # noqa: F401

from typing import TYPE_CHECKING

# 仅供静态分析（PyCharm）使用的类型提示导入：TYPE_CHECKING 恒为 False，
# 真实导入仍在函数内延迟发生（transformers 依赖链重），这样 IDE 能解析符号
# 消除「找不到模块」误报，又不会让模块导入时就背上这份开销。
if TYPE_CHECKING:
    # noqa 标注：这是给 IDE 看的类型提示，运行时不会执行
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )


def release_embedding_model():
    """
    释放 BGE-M3 embedding 模型的内存占用。

    检索链路里 embedding（2.2GB）与 reranker（2.2GB）会同时驻留，
    CPU 内存吃紧时 reranker 前向会因分配 1GB 失败（进程甚至会被系统直接杀掉、
    日志无任何 traceback）。

    【实现要点】必须**逐层把内部 torch 模块搬到 meta 设备**才能真正释放：
        BGEM3EmbeddingFunction（pymilvus 封装）
          └─ .model  ->  M3Embedder
                         └─ .model  ->  真正的 XLMRoberta 权重（约 2GB）
    注意 BGEM3EmbeddingFunction **没有 .to() 方法**（早期版本误调它，
    抛 AttributeError 被 except 吞掉，导致日志显示"已释放"但内存一点没降）。
    仅置 None 也无效——PyTorch 会缓存，必须先 offload 再 gc.collect()。

    实测：offload + gc 后进程内存从 1.89GB 降到 0.76GB（释放约 1.13GB）。
    """
    try:
        from app.lm import embedding_utils

        # 互斥：若此刻有 embedding 前向正在进行（HyDE/向量检索节点并发），
        # 跳过本次释放，避免模型被搬到 meta 后仍被 encode 触发
        # "Cannot copy out of meta tensor" 报错。宁可不释放，也不打断检索。
        if not embedding_utils._EMBED_USE_LOCK.acquire(blocking=False):
            logger.info("有 embedding 前向正在进行，跳过释放 BGE-M3（避免 meta tensor 报错）")
            return

        try:
            ef = embedding_utils._bge_m3_ef
            if ef is None:
                return

            def _offload(obj, depth: int = 0) -> None:
                """把 torch 模块的权重搬到 meta 设备（只改张量引用，不读数据）"""
                if depth > 3 or obj is None or not hasattr(obj, "to"):
                    return
                try:
                    obj.to("meta")
                except Exception:
                    return  # 非 torch 模块（如普通对象）跳过
                # 再往下一层钻
                for attr in ("model", "sentence_model", "embed_model"):
                    if hasattr(obj, attr):
                        _offload(getattr(obj, attr), depth + 1)

            # 逐层卸载：BGEM3EmbeddingFunction -> M3Embedder -> 权重
            _offload(getattr(ef, "model", None), 0)
            _offload(ef, 0)

            embedding_utils._bge_m3_ef = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("已释放 BGE-M3 embedding 模型以腾出内存")
        finally:
            embedding_utils._EMBED_USE_LOCK.release()
    except Exception as exc:
        logger.warning(f"释放 BGE-M3 模型时异常（可忽略）：{exc}")

from app.clients.reranker_config import reranker_config
from app.core.logger import logger

# 模型单例，避免重复加载（2.2GB 权重加载一次要二十几秒）
_reranker_model = None

# 模型单例的加载锁：多请求并发时避免重复加载同一份 2GB 权重
_RERANK_LOAD_LOCK = threading.Lock()

# 推理锁：torch 2.10.0+cpu 在本机做 XLMRoberta 前向计算时，
# 运行在非主线程（uvicorn 用线程池跑同步任务）会直接 Segmentation fault，
# 进程被系统杀掉、日志无任何 traceback，客户端只看到 ConnectionResetError。
# 用这把锁把模型加载与前向计算都串行化。
_RERANK_LOCK = threading.RLock()


def release_reranker_model():
    """
    释放 Reranker 模型的 2GB 权重。

    本机16GB 内存已接近枯竭（实测提交量46GB/47GB 上限），页面文件不足时
    加载任何 DLL 都会报「页面文件太小，无法完成操作」。Reranker 打分后
    立即释放，可让服务长期只驻留 embedding 的 1.9GB。
    代价是下次精排要重新加载（约 4~10 秒），可用环境变量关闭。
    """
    global _reranker_model
    if _reranker_model is None:
        return
    try:
        try:
            _reranker_model.to("meta")
        except Exception:
            pass
        _reranker_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("已释放 BGE Reranker 模型（用完即释放模式）")
    except Exception as exc:
        logger.warning(f"释放 Reranker 时异常（可忽略）：{exc}")


def get_reranker_model():
    """
    获取 BGE Reranker 模型单例（懒加载）

    :return: 已加载的 SequenceClassification 模型（eval 模式）
    :raise RuntimeError: 模型路径未配置或加载失败时抛出，由调用方降级处理
    """
    global _reranker_model
    # 加锁：并发请求可能同时进入加载分支，重复加载 2GB 权重既慢又吃内存
    with _RERANK_LOAD_LOCK:
        if _reranker_model is not None:
            return _reranker_model

        model_path = reranker_config.bge_reranker_large
        if not model_path:
            raise RuntimeError(
                "未配置 BGE_RERANKER_LARGE，请先下载 bge-reranker-large 模型并在 .env 中配置路径"
            )

        logger.info(
            f"开始初始化 BGE Reranker：path={model_path}, "
            f"device={reranker_config.bge_reranker_device}, fp16={reranker_config.bge_reranker_fp16}"
        )

        try:
            # 延迟导入 transformers：避免纯 LLM 场景白背导入开销
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # use_safetensors=False 是关键：本机走 safetensors 加载路径会段错误
            _reranker_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                dtype=torch.float32,
                use_safetensors=False,
            ).eval()

            if reranker_config.bge_reranker_device.lower().startswith("cuda"):
                _reranker_model = _reranker_model.to(reranker_config.bge_reranker_device)

            # 缓存 tokenizer，避免每次打分都重新加载
            _reranker_model.tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info("BGE Reranker 初始化成功")
        except Exception as exc:
            logger.error(f"BGE Reranker 初始化失败：{exc}")
            raise
        return _reranker_model


def rerank_scores(query: str, texts):
    """
    对「查询 vs 多个文档」批量打分

    :param query: 查询串（通常是改写后的问题）
    :param texts: 候选文档正文列表
    :return: 与 texts 一一对应的相关性得分列表（无固定取值范围，只看相对大小）
    :raise Exception: 打分失败时抛出，由调用方降级
    """
    if not query:
        raise ValueError("query 不能为空")
    if not texts:
        return []

    # 【必须整段加锁】加载（2GB 权重分配）与前向都要串行：
    #   · 模型加载本身需要分配 1GB，若此时 embedding 还占着 1.9GB 会直接 OOM；
    #   · torch 2.10.0+cpu 的 XLMRoberta 前向在并发场景下可能段错误。
    # 因此把「释放 embedding → 加载 reranker → 前向」放进同一把可重入锁，
    # 保证任意时刻只有一个 2GB 级模型在内存里，也不会被并发请求打断。
    with _RERANK_LOCK:
        # 检索阶段已结束，先腾出embedding 占的 1.9GB 再加载自己的2GB 权重
        release_embedding_model()

        model = get_reranker_model()
        tokenizer = model.tokenizer

        logger.info(f"Rerank 打分中：query 长度 {len(query)}，候选文档 {len(texts)} 条")

        # 必须用 batch 方式（FlagReranker 的做法）：
        # 把 query 重复 N 次与 N 个文档配成一批，一次前向得到 N 个分数。
        # 若逐条单独推理，模型会输出 [query位置分数, 文档位置分数] 两个值，
        # 取哪个都会拿到错误语义（取 [0] 永远是 query 自己的分数，三条结果完全相同）。
        def _forward():
            """执行一次批量前向"""
            with torch.no_grad():
                batch = tokenizer(
                    [query] * len(texts),
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                batch = {k: v.to(model.device) for k, v in batch.items()}
                return model(**batch).logits.view(-1).tolist()

        try:
            logits = _forward()
        except RuntimeError as exc:
            # 内存不足时的最后兜底：再释放一次并重试
            msg = str(exc)
            if "not enough memory" not in msg and "alloc_cpu" not in msg:
                raise
            logger.warning(f"Reranker 内存不足，重试一次：{msg}")
            release_embedding_model()
            gc.collect()
            logits = _forward()

        # 【可选：用完即释放】
        # P2-1：默认常驻（RERANKER_RELEASE_AFTER_SCORE=0），避免每次精排都重复
        # 加载 2GB 权重（约 4~10 秒），换取更稳定的精排响应。
        # 本机内存紧张（16GB、页面文件不足）时，可在 .env 设
        # RERANKER_RELEASE_AFTER_SCORE=1 恢复「打分后立即释放」。
        if os.environ.get("RERANKER_RELEASE_AFTER_SCORE", "0") not in ("0", "false", "False"):
            release_reranker_model()

    scores = [float(s) for s in logits]
    logger.info(f"Rerank 打分完成，最高分 {max(scores):.4f}")
    return scores


if __name__ == "__main__":
    if not reranker_config.bge_reranker_large:
        print("未配置 BGE_RERANKER_LARGE，跳过自测")
    else:
        demo_docs = [
            "HAK 180 烫金机的操作面板位于机器正前方，开启电源后先设置温度",
            "苏泊尔电磁炉采用线圈式电磁加热技术，功率可达5000W",
            "本文档介绍烫金机的日常保养与清洁方法",
        ]
        q = "HAK 180 烫金机怎么操作？"
        sc = rerank_scores(q, demo_docs)
        print(f"\n查询: {q}")
        for idx in sorted(range(len(sc)), key=lambda i: sc[i], reverse=True):
            print(f"  [{sc[idx]:>8.4f}] {demo_docs[idx][:40]}")
        print("\n期望：烫金机相关文档得分高于苏泊尔电磁炉")