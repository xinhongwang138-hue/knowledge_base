"""
MinIO 对象存储配置

只负责「配置从哪来、怎么校验」，不含业务逻辑。
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
class MinIOConfig:
    """MinIO 配置（均可通过 .env 覆盖）"""

    # 客户端连接用地址（不带协议前缀）
    # 本机直跑：127.0.0.1:9000；容器内：必须用服务名 minio:9000，
    # 否则 127.0.0.1 指向容器自身，上传/读取全部失败
    endpoint: str = field(default="")
    # 浏览器/前端访问用地址（不带协议前缀）
    # 容器场景下 endpoint=minio:9000，浏览器解析不到该域名，
    # 故预签名 URL 必须用对外地址生成；未配置时回退为 endpoint
    public_endpoint: str = field(default="")
    # 预签名的对外协议是否 HTTPS（由 public_endpoint 的协议前缀推导，默认同 minio_secure）
    public_secure: bool = field(default=False)
    # 预签名签名使用的 region。固定值可避免每次预签都向服务端查询桶归属地
    # （MinIO 默认 us-east-1）
    region: str = field(default="")
    access_key: str = field(default="")
    secret_key: str = field(default="")
    # 存储桶名
    bucket_name: str = field(default="knowledge-base-files")
    # 图片在桶内的目录前缀，形如 /upload-images
    minio_img_dir: str = field(default="upload-images")
    # 是否启用 HTTPS（MinIO 本地默认 HTTP，云端通常 HTTPS）
    minio_secure: bool = field(default=False)
    # 预签名 URL 有效期（秒），通过 .env 的 MINIO_PRESIGNED_EXPIRY 配置
    # （单位：天，支持小数，如 0.5 = 12 小时），未配置默认 7 天
    presigned_expiry_seconds: int = field(default=7 * 24 * 3600)

    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = os.getenv("MINIO_ENDPOINT", "")
        if not self.public_endpoint:
            self.public_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT", "")
        if not self.region:
            self.region = os.getenv("MINIO_REGION", "") or "us-east-1"
        if not self.access_key:
            self.access_key = os.getenv("MINIO_ACCESS_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("MINIO_SECRET_KEY", "")
        if not self.bucket_name:
            self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "") or "knowledge-base-files"
        if not self.minio_img_dir:
            self.minio_img_dir = os.getenv("MINIO_IMG_DIR", "") or "upload-images"

        # 预签名有效期：.env 的 MINIO_PRESIGNED_EXPIRY 以「天」为单位（支持小数），
        # 内部统一换算为秒；未配置或解析失败时保持默认 7 天
        try:
            raw_expiry = os.getenv("MINIO_PRESIGNED_EXPIRY", "").strip()
            if raw_expiry:
                self.presigned_expiry_seconds = int(float(raw_expiry) * 24 * 3600)
        except (TypeError, ValueError):
            pass  # 非法值按默认 7 天处理，避免因配置失误导致服务异常

        # 前缀统一去掉首尾斜杠，避免拼接时出现 //upload-images//xxx
        self.minio_img_dir = self.minio_img_dir.strip("/")
        self.endpoint = self.endpoint.strip().rstrip("/")

        # endpoint 可能带了 http:// 前缀，MinIO 客户端要求不带协议，这里剥掉
        for prefix in ("http://", "https://"):
            if self.endpoint.startswith(prefix):
                self.endpoint = self.endpoint[len(prefix):]
                self.minio_secure = prefix == "https://"
                break

        # 对外地址同样剥掉协议前缀，并按前缀推导对外协议
        self.public_endpoint = (self.public_endpoint or "").strip().rstrip("/")
        self.public_secure = self.minio_secure
        for prefix in ("http://", "https://"):
            if self.public_endpoint.startswith(prefix):
                self.public_endpoint = self.public_endpoint[len(prefix):]
                self.public_secure = prefix == "https://"
                break
        # 未单独配置对外地址时回退为连接地址（本机直跑场景二者一致）
        if not self.public_endpoint:
            self.public_endpoint = self.endpoint

    @property
    def public_base_url(self) -> str:
        """图片对外可访问的基础地址（协议 + 对外端点）

        注意：必须用 public_endpoint 而非 endpoint。
        容器内 endpoint 是服务名 minio:9000，浏览器无法解析，
        直接拼出来的图片地址会 100% 加载失败。
        """
        scheme = "https" if self.public_secure else "http"
        return f"{scheme}://{self.public_endpoint}"


# 全局单例
minio_config = MinIOConfig()


if __name__ == "__main__":
    print(f"endpoint       = {minio_config.endpoint or '(未配置)'}")
    print(f"access_key     = {minio_config.access_key or '(未配置)'}")
    print(f"bucket_name    = {minio_config.bucket_name}")
    print(f"minio_img_dir  = {minio_config.minio_img_dir}")
    print(f"minio_secure   = {minio_config.minio_secure}")
    print(f"public_base_url= {minio_config.public_base_url}")
