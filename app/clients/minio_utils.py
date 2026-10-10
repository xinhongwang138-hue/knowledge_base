"""
MinIO 客户端工具类

封装客户端初始化与桶管理，采用模块级单例，避免重复建立连接。
"""
import json

from minio import Minio

from app.clients.minio_config import minio_config
from app.core.logger import logger

# 全局MinIO 客户端实例（懒加载单例）
_minio_client = None


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
    )


def _set_bucket_policy(bucket_name: str) -> str:
    """
    生成桶访问策略（JSON）——允许所有用户读取桶内对象，
    这样图片上传后能被 MD 里的在线 URL 直接访问

    :param bucket_name: 目标桶名
    :return: 序列化后的策略字符串
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    return json.dumps(policy)


def _create_bucket_ready(client: Minio) -> None:
    """
    检查桶是否存在，不存在则创建并设置公开读取策略
    """
    bucket_name = minio_config.bucket_name
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
            logger.info(f"MinIO 桶 {bucket_name} 已创建，并设置公开读取策略")
        else:
            logger.info(f"MinIO 桶 {bucket_name} 已存在，无需重复创建")
    except Exception as exc:
        # 桶初始化失败不直接抛，让调用方决定是否继续
        logger.warning(f"MinIO 桶初始化失败（若桶已存在可忽略）：{exc}")


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
                f"-e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin "
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