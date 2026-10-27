"""
MinIO 客户端工具类

封装客户端初始化与桶管理，采用模块级单例，避免重复建立连接。
"""
from datetime import timedelta
from typing import Optional
from urllib.parse import unquote, urlsplit

from minio import Minio

from app.clients.minio_config import minio_config
from app.core.logger import logger

# 全局MinIO 客户端实例（懒加载单例）
_minio_client = None
# 预签名专用客户端（用对外地址构造，仅本地签名计算）
_presign_client = None


def _create_minio_client() -> Minio:
    """
    创建 MinIO 客户端连接
    :return: 初始化完成的 MinIO 客户端对象
    """
    return Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
        # 固定 region，避免 SDK 为取 bucket location 额外发一次 HTTP 请求
        region=minio_config.region,
    )


def get_presign_client() -> Minio:
    """获取「预签名专用」客户端（懒加载）。

    为什么需要独立客户端：SigV4 预签名把 Host 计入签名。
    容器内 endpoint 是服务名 minio:9000，用它签出来的 URL
    浏览器既解析不到该域名、签名也对不上，图片必然 403/无法加载。
    因此单独用对外地址（MINIO_PUBLIC_ENDPOINT）构造客户端，
    该客户端只做本地签名运算，不建立真实连接（region 已固定）。

    :return: 使用对外地址的 MinIO 客户端
    """
    global _presign_client
    if _presign_client is None:
        _presign_client = Minio(
            endpoint=minio_config.public_endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.public_secure,
            region=minio_config.region,
        )
        logger.info(
            f"预签名客户端已初始化：{minio_config.public_endpoint}"
            f"（客户端连接地址：{minio_config.endpoint}）"
        )
    return _presign_client


def _create_bucket_ready(client: Minio) -> None:
    """
    检查桶是否存在，不存在则创建（默认私有桶，不设置公开读取策略）。

    P1-5：不再对桶设置 Principal "*" 公开读策略——对象访问一律走预签名 URL
    （见 get_presigned_object_url），避免知识库文件被匿名公开拉取。
    若桶已存在且为历史公开桶，此处保持不动以兼容存量数据，需要收紧时由运维
    在 MinIO 控制台手动移除策略。
    """
    bucket_name = minio_config.bucket_name
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"MinIO 桶 {bucket_name} 已创建（默认私有，不设置公开读策略）")
        else:
            logger.info(f"MinIO 桶 {bucket_name} 已存在，无需重复创建")
    except Exception as exc:
        # 桶初始化失败不直接抛，让调用方决定是否继续
        logger.warning(f"MinIO 桶初始化失败（若桶已存在可忽略）：{exc}")


def get_presigned_object_url(
    client: Minio,
    bucket_name: str,
    object_name: str,
    expires_seconds: Optional[int] = None,
) -> str:
    """
    生成对象预签名 URL（私有桶场景下 MD 内图片的正常访问方式）。

    预签名 URL 在有效期内无需凭据即可下载，过期后需重新生成；
    有效期默认取 .env 配置 MINIO_PRESIGNED_EXPIRY（默认 7 天），
    调用方可显式覆盖。

    :param client: MinIO 客户端
    :param bucket_name: 桶名
    :param object_name: 对象名
    :param expires_seconds: 有效期秒数，None 表示使用配置默认值（默认 7 天）
    :return: 预签名 URL；生成失败时返回空串（由调用方决定兜底策略）
    """
    if expires_seconds is None:
        expires_seconds = minio_config.presigned_expiry_seconds
    try:
        # 统一用「对外地址」的客户端签名：Host 参与 SigV4 签名，
        # 用容器内地址 minio:9000 签出来的 URL 浏览器解析不到、签名也验不过。
        # client 参数保留以兼容既有调用方，实际不再使用其 endpoint。
        url = get_presign_client().presigned_get_object(
            bucket_name,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )
        logger.debug(f"已生成预签名 URL：{object_name}（{expires_seconds}s）")
        return url
    except Exception as exc:
        logger.warning(
            f"生成预签名 URL 失败（bucket={bucket_name}, object={object_name}）：{exc}"
        )
        return ""


def presign_public_image_url(public_url: str, expires_seconds: Optional[int] = None) -> str:
    """
    将「公开形态的图片 URL」实时重签为预签名 URL。

    用于私有桶改造后，把历史直链（public_base_url/bucket/object）或
    已过期的预签名 URL 重新签名，保证查询链路对外返回的图片可访问。

    仅处理属于本 MinIO 桶的 URL（host:port 匹配且路径含 bucket 前缀），
    非本桶 URL（外链、web 来源）原样返回，避免误重签。

    :param public_url: 原始图片 URL（公开直链或已过期的预签名 URL）
    :param expires_seconds: 有效期秒数，None 使用配置默认值
    :return: 预签名 URL；非本桶 URL 原样返回；重签失败时返回空串
    """
    if not public_url:
        return public_url or ""

    try:
        parts = urlsplit(public_url)
        # host 校验：本 MinIO 的「对外地址」与「连接地址」都算本桶，
        # 兼容历史数据里两种写法（127.0.0.1:9000 与 minio:9000）
        allowed_hosts = {
            minio_config.public_endpoint.split(":")[0].lower(),
            minio_config.endpoint.split(":")[0].lower(),
        }
        url_host = (parts.hostname or "").lower()
        if url_host not in allowed_hosts:
            return public_url  # 外链不处理

        # 路径解析：/bucket/object_name，需以本桶前缀开头
        path = unquote(parts.path or "").strip("/")
        bucket = minio_config.bucket_name
        if not path.startswith(bucket + "/"):
            return public_url  # 非本桶对象不处理
        object_name = path[len(bucket) + 1:]
    except Exception as exc:
        logger.warning(f"解析图片 URL 失败（{public_url}）：{exc}")
        return ""

    return get_presigned_object_url(
        get_minio_client(), minio_config.bucket_name, object_name, expires_seconds
    )


def get_minio_client() -> Minio:
    """
    获取全局 MinIO 客户端（懒加载）

    :return: 全局唯一的 MinIO 客户端对象
    :raise ValueError: 配置缺失时抛出，附带申请/排查指引
    """
    global _minio_client

    if _minio_client is None:
        missing = [
            name
            for name, val in (
                ("MINIO_ENDPOINT", minio_config.endpoint),
                ("MINIO_ACCESS_KEY", minio_config.access_key),
                ("MINIO_SECRET_KEY", minio_config.secret_key),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"[MinIO] 配置缺失：请在 .env 中配置 {'、'.join(missing)}\n"
                f"  本地起一个：docker run -d --name minio -p 9000:9000:9001 "
                f"-e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=<见 .env 配置> "
                f"quay.io/minio/minio server /data --console-address \":9001\""
            )

        logger.info("开始初始化 MinIO 客户端（首次调用，懒加载）")
        client = _create_minio_client()
        _create_bucket_ready(client)
        _minio_client = client
        logger.info("MinIO 客户端初始化完成，已就绪")

    return _minio_client


if __name__ == "__main__":
    logger.info("===== 开始 MinIO 客户端测试 =====")
    try:
        c = get_minio_client()
        exists = c.bucket_exists(minio_config.bucket_name)
        logger.info(f"桶 {minio_config.bucket_name} 存在：{exists}")
        logger.info(f"公开地址前缀：{minio_config.public_base_url}")
    except Exception as exc:
        logger.error(f"❌ MinIO 测试失败：{exc}")
    finally:
        logger.info("===== MinIO 客户端测试结束 =====")