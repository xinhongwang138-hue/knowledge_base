"""
BGE-M3 Embedding 配置

适配 BGE-M3 的模型路径、运行设备、半精度开关，配置风格与 lm_config / minio_config 一致。
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
class EmbeddingConfig:
    """BGE-M3 配置（均可通过 .env 覆盖）"""

    # 本地模型目录，指向 ModelScope/HuggingFace 下载好的模型文件
    bge_m3_path: str = field(default="")
    # 模型仓库标识，线上部署可直接填 "BAAI/bge-m3" 由框架自动拉取
    bge_m3: str = field(default="BAAI/bge-m3")
    # 运行设备：cuda:0 / cpu
    bge_device: str = field(default="cpu")
    # 是否开启 FP16 半精度推理（仅 GPU 有效）
    bge_fp16: bool = field(default=False)

    def __post_init__(self):
        if not self.bge_m3_path:
            self.bge_m3_path = os.getenv("BGE_M3_PATH", "") or ""
        if not self.bge_m3:
            self.bge_m3 = os.getenv("BGE_M3", "") or "BAAI/bge-m3"
        if not self.bge_device:
            self.bge_device = os.getenv("BGE_DEVICE", "") or "cpu"
        # 半精度单独用 _env_bool 解析，兼容 "1"/"true"/"True" 等写法
        self.bge_fp16 = _env_bool("BGE_FP16", self.bge_fp16)

        # 本地路径去掉首尾空白、引号与「尾部」斜杠（仅 rstrip，务必保留 POSIX 绝对路径开头的 /，否则 Linux 容器内路径会被误判为相对路径）
        self.bge_m3_path = self.bge_m3_path.strip().strip('"').strip("'").rstrip("/")
        self.bge_device = self.bge_device.strip().strip('"').strip("'")

        # 重要：CPU 环境不支持 FP16 半精度，强制关闭，否则模型推理会报错
        if self.bge_device.lower().startswith("cpu"):
            if self.bge_fp16:
                logger_fp16 = "检测到 CPU 设备，已自动关闭 BGE_FP16（CPU 不支持半精度）"
                print(logger_fp16)
            self.bge_fp16 = False


# 全局单例
embedding_config = EmbeddingConfig()


if __name__ == "__main__":
    print(f"bge_m3_path  = {embedding_config.bge_m3_path or '(未配置，将按仓库标识自动下载)'}")
    print(f"bge_m3       = {embedding_config.bge_m3}")
    print(f"bge_device   = {embedding_config.bge_device}")
    print(f"bge_fp16     = {embedding_config.bge_fp16}")
