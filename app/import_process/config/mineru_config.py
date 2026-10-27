"""
MinerU 解析服务配置

只负责「配置从哪来、怎么校验」，不含任何业务逻辑。
调用方（节点）通过 `mineru_config` 单例获取配置。
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")

# MinerU API v4 根地址（.env 未配置时的兜底值）
_DEFAULT_API_ROOT = "https://mineru.net/api/v4"


def _normalize_api_root(raw: str) -> str:
    """
    把 .env 里可能写法的 URL 规整成 API 根地址（不含末尾斜杠）

    兼容三种写法：
        "https://mineru.net/api/v4"                      -> 原样
        "https://mineru.net/api/v4/"                     -> 去掉末尾斜杠
        "https://mineru.net/api/v4/extract/task"         -> 截断到 /api/v4
        "https://mineru.net/api/v4/extract-results/batch/x" -> 截断到 /api/v4
    """
    url = (raw or "").strip().strip('"').strip("'").rstrip("/")
    if not url:
        return _DEFAULT_API_ROOT

    # 找到 /api/v4 这一段，在它之后截断——之后的部分都是具体接口路径
    marker = "/api/v4"
    idx = url.find(marker)
    if idx != -1:
        return url[: idx + len(marker)]
    return url


def _env_int(key: str, default: int, min_value: int = 1) -> int:
    """
    读取整数型环境变量，非法或缺失时回退默认值

    刻意不放宽成"能转就转"：写错配置时立刻看到默认值，
    比悄悄用了一个奇怪的数字更容易排查。
    """
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw.strip().strip('"').strip("'"))
    except (TypeError, ValueError):
        return default
    return max(value, min_value)


@dataclass
class MineruConfig:
    """MinerU API 配置（全部字段均可通过 .env 覆盖，字段留空则用默认值）"""

    api_root: str = field(default="")
    api_token: str = field(default="")
    # 单个文件大小上限 200MB，超限服务端会拒
    max_file_mb: int = 0
    # 轮询间隔（秒）
    poll_interval: int = 0
    # 单个任务最长等待时间（秒），超时放弃，避免无限轮询
    timeout: int = 0

    def __post_init__(self):
        # 优先读 MINERU_API_ROOT，其次兼容 MINERU_BASE_URL
        # （旧配置名，保留兼容以免换环境时报错）
        raw = self.api_root or os.getenv("MINERU_API_ROOT", "") or os.getenv(
            "MINERU_BASE_URL", ""
        )
        self.api_root = _normalize_api_root(raw)

        if not self.api_token:
            self.api_token = os.getenv("MINERU_API_TOKEN", "")

        # 以下三项：显式传参优先，否则读 .env，都没有才用默认值
        self.max_file_mb = self.max_file_mb or _env_int("MINERU_MAX_FILE_MB", 200)
        self.poll_interval = self.poll_interval or _env_int("MINERU_POLL_INTERVAL", 1)
        self.timeout = self.timeout or _env_int("MINERU_TIMEOUT", 1800)

        # 配置缺失要在启动时就报错，而不是等到调API 才崩
        if not self.api_token:
            raise ValueError(
                "缺少 MinerU API Token，请检查 .env 中是否配置 MINERU_API_TOKEN"
            )

    # ---------- 各接口的完整 URL ----------

    @property
    def batch_submit_url(self) -> str:
        """申请本地文件上传链接"""
        return f"{self.api_root}/file-urls/batch"

    @property
    def batch_result_url(self) -> str:
        """
        查询批量任务结果的前缀（调用时需拼上 batch_id）

        实际请求地址形如：
            https://mineru.net/api/v4/extract-results/batch/{batch_id}
        """
        return f"{self.api_root}/extract-results/batch"

    def result_url(self, batch_id: str) -> str:
        """按 batch_id 拼出完整的轮询地址"""
        return f"{self.batch_result_url}/{batch_id}"

    @property
    def auth_headers(self) -> dict:
        """请求头，认证用 Bearer"""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }


# 全局单例：导入即完成配置校验，配置有问题会在启动时立刻暴露
mineru_config = MineruConfig()


if __name__ == "__main__":
    print(f"api_root      = {mineru_config.api_root}")
    print(f"token前 8 位   = {mineru_config.api_token[:8]}...")
    print(f"上传链接接口    = {mineru_config.batch_submit_url}")
    print(f"结果查询接口    = {mineru_config.batch_result_url}")
    print(f"max_file_mb   = {mineru_config.max_file_mb}")
    print(f"poll_interval = {mineru_config.poll_interval}")
    print(f"timeout       = {mineru_config.timeout}")
