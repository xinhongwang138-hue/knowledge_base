"""
BGE Reranker 重排序模型配置

Reranker 与 Embedding（bge-m3）是两种不同的模型：
    - Embedding：把文本编码成向量，用于「快速粗筛」
    - Reranker：拿「问题 + 文档」一起进模型精读打分，用于「慢速精排」
两者不能互换，各自有独立的模型路径与运行参数。
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().strip('"').strip("'").lower() in ("true", "1", "yes", "on")


@dataclass
class RerankerConfig:
    """BGE Reranker 配置（均可通过 .env 覆盖）"""

    # 通用模型目录：优先读 BGE_RERANKER_PATH（base / large 都适用），
    # 未配置时回退到历史变量 BGE_RERANKER_LARGE，避免旧 .env 失效
    bge_reranker_path: str = field(default="")
    # 本地模型目录（历史字段，保留兼容）
    bge_reranker_large: str = field(default="")
    # 运行设备：cpu / cuda:0
    bge_reranker_device: str = field(default="cpu")
    # 是否开启 FP16 半精度（仅 GPU 有效）
    bge_reranker_fp16: bool = field(default=False)

    def __post_init__(self):
        path = os.getenv("BGE_RERANKER_PATH", "") or self.bge_reranker_path
        if not path:
            path = os.getenv("BGE_RERANKER_LARGE", "") or self.bge_reranker_large
        self.bge_reranker_path = (path or "").strip().strip('"').strip("'").rstrip("/")
        # 兼容旧代码里对 bge_reranker_large 的直接引用
        self.bge_reranker_large = self.bge_reranker_path
        if not self.bge_reranker_device:
            self.bge_reranker_device = os.getenv("BGE_RERANKER_DEVICE", "") or "cpu"
        self.bge_reranker_fp16 = _env_bool("BGE_RERANKER_FP16", self.bge_reranker_fp16)

        self.bge_reranker_path = (
            self.bge_reranker_path.strip().strip('"').strip("'").rstrip("/")
        )
        self.bge_reranker_large = self.bge_reranker_path
        self.bge_reranker_device = self.bge_reranker_device.strip().strip('"').strip("'")

        # CPU 环境不支持 FP16 半精度，强制关闭，否则推理会报错
        if self.bge_reranker_device.lower().startswith("cpu"):
            self.bge_reranker_fp16 = False


# 全局单例
reranker_config = RerankerConfig()


if __name__ == "__main__":
    print(f"bge_reranker_large  = {reranker_config.bge_reranker_large or '(未配置)'}")
    print(f"bge_reranker_device = {reranker_config.bge_reranker_device}")
    print(f"bge_reranker_fp16   = {reranker_config.bge_reranker_fp16}")
    print(f"模型目录存在: {os.path.isdir(reranker_config.bge_reranker_large)}")
