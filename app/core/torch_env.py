"""
PyTorch / MKL 运行时环境初始化（必须在任何 torch 相关 import 之前执行）

为什么需要这个模块：
    torch 2.10.0+cpu 捆绑 Intel oneMKL 2025.3，在部分 Intel 混合架构 CPU
    （本机为 Alder Lake，含 E-core，MKL 默认按物理核开 12 线程）上执行
    BGE-M3 前向计算时会直接 Segmentation fault（Windows 退出码 139，
    无 Python traceback）。

排查结论：
    - 与 attention 实现无关（eager / sdpa 均崩）
    - 与权重完整性无关（pytorch_model.bin 大小与官方一致）
    - 与 FlagEmbedding / pymilvus 封装无关（transformers 裸接口同样崩）
    - torch.set_num_threads() 在 torch 已初始化后调用【无效】
    - 以 MKL_NUM_THREADS=1 启动进程即可正常运行（不要设 OMP_NUM_THREADS，
      否则会连带影响 OpenBLAS，导致 numpy 报内存分配失败）
    => 根因是 MKL 在 E-core 上初始化多线程计算内核时崩溃

为什么放在core 而不是 embedding_utils：
    `import torch` 一旦发生，相关环境变量就来不及生效了。
    而 app 里多个模块（pymilvus 相关、transformers 相关）都可能间接把 torch 拉进来，
    若只在 embedding_utils 里设，可能已经太晚。本模块被设计为「应用最早导入的模块」，
    由 main_graph / 入口脚本第一行导入，确保早于任何 torch 初始化。

用法：
    import app.core.torch_env  # noqa: F401  仅导入即生效，放在程序最开头
"""
import os

# 用 setdefault：外部若已显式配置则尊重外部值（如有 GPU 想放开多线程）
#
# 【重要】只限 MKL，不要设 OMP_NUM_THREADS。
# 设了 OMP_NUM_THREADS=1 会连带影响 OpenBLAS，导致 numpy / BLAS 在做
# 矩阵运算时报「OpenBLAS error: Memory allocation still failed after 10 retries」
# 而完全无法工作；而真正需要限制的只有 Intel oneMKL 那一条路径。
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 已设置完成，供调用方判断是否为本模块首次生效
TORCH_ENV_INITIALIZED = True


if __name__ == "__main__":
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"), "(刻意不设，避免影响 OpenBLAS)")