"""
MinIO 对象存储配置

只负责「配置从哪来、怎么校验」，不含业务逻辑。
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 用绝对路径加载 .env，避免依赖运行时的工作目录
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().strip('"').strip("'").lower() in ("true", "1", "yes", "on")


@dataclass
class MinIOConfig:
    """MinIO 配置（均可通过 .env 覆盖）"""

    # 服务地址，形如 47.94.86.115:9000（不带协议前缀）
    endpoint: str = field(default="")
    access_key: str = field(default="")
    secret_key: str = field(default="")
    # 存储桶名
    bucket_name: str = field(default="knowledge-base-files")
    # 图片在桶内的目录前缀，形如 /upload-images
    minio_img_dir: str = field(default="upload-images")
    # 是否启用 HTTPS（MinIO 本地默认 HTTP，云端通常 HTTPS）
    minio_secure: bool = field(default=False)

    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = os.getenv("MINIO_ENDPOINT", "")
        if not self.access_key:
            self.access_key = os.getenv("MINIO_ACCESS_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("MINIO_SECRET_KEY", "")
        if not self.bucket_name:
            self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "") or "knowledge-base-files"
        if not self.minio_img_dir:
            self.minio_img_dir = os.getenv("MINIO_IMG_DIR", "") or "upload-images"

        # 前缀统一去掉首尾斜杠，避免拼接时出现 //upload-images//xxx
        self.minio_img_dir = self.minio_img_dir.strip("/")
        self.endpoint = self.endpoint.strip().rstrip("/")

        # endpoint 可能带了 http:// 前缀，MinIO 客户端要求不带协议，这里剥掉
        for prefix in ("http://", "https://"):
            if self.endpoint.startswith(prefix):
                self.endpoint = self.endpoint[len(prefix):]
                self.minio_secure = prefix == "https://"
                break

    @property
    def public_base_url(self) -> str:
        """图片对外可访问的基础地址（协议 + 端点）"""
        scheme = "https" if self.minio_secure else "http"
        return f"{scheme}://{self.endpoint}"


# 全局单例
minio_config = MinIOConfig()


if __name__ == "__main__":
    print(f"endpoint       = {minio_config.endpoint or '(未配置)'}")
    print(f"access_key     = {minio_config.access_key or '(未配置)'}")
    print(f"bucket_name    = {minio_config.bucket_name}")
    print(f"minio_img_dir  = {minio_config.minio_img_dir}")
    print(f"minio_secure   = {minio_config.minio_secure}")
    print(f"public_base_url= {minio_config.public_base_url}")