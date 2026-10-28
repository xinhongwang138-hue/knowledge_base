"""
Milvus 向量数据库配置

只负责「配置从哪来、怎么兜底」，不含业务逻辑。
配置风格与 lm_config / minio_config 保持一致。
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MilvusConfig:
    """Milvus 配置（均可通过 .env 覆盖）"""

    # Milvus 服务地址，形如 http://127.0.0.1:19530
    milvus_url: str = field(default="")
    # 存储文本切片的集合名
    # 默认留空，交给 __post_init__ 从 .env 读取后再兜底；
    # 若这里写死默认名，__post_init__ 的 if not self.xxx 永远为 False，env 配置将静默失效
    chunks_collection: str = field(default="")
    # 预留：实体名称集合
    entity_name_collection: str = field(default="")
    # 存储「文档 -> 主体名称」的集合名
    item_name_collection: str = field(default="")

    def __post_init__(self):
        if not self.milvus_url:
            self.milvus_url = os.getenv("MILVUS_URL", "") or "http://127.0.0.1:19530"
        if not self.chunks_collection:
            self.chunks_collection = os.getenv("CHUNKS_COLLECTION", "") or "kb_chunks"
        if not self.entity_name_collection:
            self.entity_name_collection = (
                os.getenv("ENTITY_NAME_COLLECTION", "") or "kb_graph_entity_names"
            )
        if not self.item_name_collection:
            self.item_name_collection = os.getenv("ITEM_NAME_COLLECTION", "") or "kb_item_names"

        # 地址统一去掉尾部斜杠，避免与其他 URL 拼接时出现双斜杠
        self.milvus_url = self.milvus_url.strip().rstrip("/")


# 全局单例
milvus_config = MilvusConfig()


if __name__ == "__main__":
    print(f"milvus_url              = {milvus_config.milvus_url}")
    print(f"chunks_collection       = {milvus_config.chunks_collection}")
    print(f"entity_name_collection  = {milvus_config.entity_name_collection}")
    print(f"item_name_collection    = {milvus_config.item_name_collection}")
