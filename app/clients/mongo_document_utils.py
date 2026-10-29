"""
已入库文档台账（MongoDB）

为什么单独维护一张表，而不是每次去 Milvus 里 group by：
    · Milvus 是向量库，按 file_title 聚合要全量扫切片，文档一多就是灾难；
    · 切片本身没有「这份文档什么时候导进来的、一共几条、主体是谁」这类元信息，
      全靠扫 content 反推；
    · 删除文档需要先知道它有哪些切片、图片落在 MinIO 哪个前缀下，台账正好承担这个索引角色。

数据结构（集合 kb_documents，(tenant_id, doc_id) 唯一）：
    {
        "doc_id":      文件级幂等键，与 Milvus 切片的 doc_id 同源（sha1(tenant|file_title) 派生）
        "tenant_id":   所属租户
        "file_title":  文件名（去后缀），同时是 MinIO 图片目录名
        "item_name":   LLM 识别出的主体名
        "chunk_count": 切片条数
        "task_id":     导入任务 ID（手动同步进来的记录为空）
        "imported_at": 导入时间戳（秒）；从 Milvus 回填的老数据为 None
        "updated_at":  台账最后一次更新时间戳（秒）
    }

写入时机：
    · 导入流程跑完后由导入服务写入（run_graph_task → _record_document）；
    · 台账为空时（升级前入库的老数据）由 GET /documents 触发一次 Milvus 回填。
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.logger import PROJECT_ROOT, logger

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")

DOC_COLLECTION_NAME = "kb_documents"


class DocumentMongoTool:
    """文档台账的连接与集合初始化（与历史记录工具同构，但独立负责一张表）"""

    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL")
        self.db_name = os.getenv("MONGO_DB_NAME")
        if not self.mongo_url or not self.db_name:
            raise ValueError("MongoDB 配置缺失：请在 .env 中配置 MONGO_URL 与 MONGO_DB_NAME")

        self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.db_name]
        self.documents = self.db[DOC_COLLECTION_NAME]

        # 唯一键：同一租户下同一份文件只有一条台账（重导时覆盖，不追加）
        try:
            self.documents.create_index(
                [("tenant_id", ASCENDING), ("doc_id", ASCENDING)], unique=True
            )
        except Exception as exc:
            logger.warning(f"文档台账唯一索引创建失败，降级为普通索引：{exc}")
            self.documents.create_index([("tenant_id", ASCENDING), ("doc_id", ASCENDING)])
        # 列表按导入时间倒序，走这个索引避免内存排序
        self.documents.create_index([("tenant_id", ASCENDING), ("imported_at", DESCENDING)])

        logger.info(f"成功连接到 MongoDB：{self.db_name}.{DOC_COLLECTION_NAME}")

    def ping(self) -> bool:
        """探活：确认数据库真正可读写"""
        try:
            self.client.admin.command("ping")
            return True
        except Exception as exc:
            logger.error(f"文档台账探活失败：{exc}")
            return False


# 全局单例：避免重复建立连接
_document_mongo_tool: Optional[DocumentMongoTool] = None


def get_document_mongo_tool() -> DocumentMongoTool:
    """
    获取台账工具单例（懒加载）

    :return: DocumentMongoTool 单例
    """
    global _document_mongo_tool
    if _document_mongo_tool is None:
        _document_mongo_tool = DocumentMongoTool()
    return _document_mongo_tool


def upsert_document(
    doc_id: str,
    tenant_id: str,
    file_title: str,
    item_name: str = "",
    chunk_count: int = 0,
    task_id: str = "",
    imported_at: Optional[float] = None,
    source: str = "import",
) -> Dict[str, Any]:
    """
    写入或更新一条文档台账

    :param doc_id: 文件级幂等键（与 Milvus 切片的 doc_id 一致）
    :param tenant_id: 所属租户
    :param file_title: 文件名（去后缀）
    :param item_name: 主体名
    :param chunk_count: 切片条数
    :param task_id: 导入任务 ID（回填时为空）
    :param imported_at: 导入时间戳；回填老数据时传 None
    :param source: import=导入流程写入 / sync=从 Milvus 回填
    :return: 落库的文档摘要；参数不全时返回空字典
    """
    if not doc_id or not tenant_id:
        logger.warning(f"台账写入缺少关键字段：doc_id={doc_id!r}, tenant_id={tenant_id!r}")
        return {}

    now = datetime.now().timestamp()
    document = {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "file_title": file_title or "",
        "item_name": item_name or "",
        "chunk_count": int(chunk_count or 0),
        "task_id": task_id or "",
        "imported_at": imported_at,
        "updated_at": now,
    }
    try:
        tool = get_document_mongo_tool()
        tool.documents.update_one(
            {"tenant_id": tenant_id, "doc_id": doc_id},
            {"$set": document},
            upsert=True,
        )
        logger.info(
            f"台账已更新：{file_title}（tenant={tenant_id}, doc_id={doc_id[:8]}…, "
            f"chunks={chunk_count}, source={source}）"
        )
        return document
    except Exception as exc:
        # 台账写失败不该让导入流程失败：文档已经在 Milvus 里，缺一条索引还能靠 sync 补回来
        logger.error(f"台账写入失败（不影响导入结果，可用 /documents/sync 补）：{exc}")
        return {}


def list_documents(tenant_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    列出某租户的已入库文档，按导入时间倒序

    :param tenant_id: 租户标识
    :param limit: 返回条数上限
    :return: 文档列表；查询失败返回空列表
    """
    if not tenant_id:
        return []
    try:
        tool = get_document_mongo_tool()
        cursor = (
            tool.documents.find({"tenant_id": tenant_id})
            # 回填的老数据 imported_at 为 None，排在最后而不是报错
            .sort([("imported_at", DESCENDING), ("updated_at", DESCENDING)])
            .limit(max(1, limit))
        )
        return [
            {
                "doc_id": doc.get("doc_id", ""),
                "tenant_id": doc.get("tenant_id", ""),
                "file_title": doc.get("file_title", ""),
                "item_name": doc.get("item_name", ""),
                "chunk_count": doc.get("chunk_count", 0),
                "task_id": doc.get("task_id", ""),
                "imported_at": doc.get("imported_at"),
                "updated_at": doc.get("updated_at"),
            }
            for doc in cursor
        ]
    except Exception as exc:
        logger.error(f"查询文档台账失败：{exc}")
        return []


def get_document(doc_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    取单条台账（删除前用它拿 file_title 定位 MinIO 前缀）

    :param doc_id: 文件级幂等键
    :param tenant_id: 租户标识
    :return: 台账记录；不存在返回 None
    """
    if not doc_id or not tenant_id:
        return None
    try:
        tool = get_document_mongo_tool()
        return tool.documents.find_one({"doc_id": doc_id, "tenant_id": tenant_id})
    except Exception as exc:
        logger.error(f"查询文档台账失败：{exc}")
        return None


def delete_document(doc_id: str, tenant_id: str) -> int:
    """
    删除一条台账记录

    :param doc_id: 文件级幂等键
    :param tenant_id: 租户标识
    :return: 实际删除条数；失败返回 0
    """
    try:
        tool = get_document_mongo_tool()
        result = tool.documents.delete_many({"doc_id": doc_id, "tenant_id": tenant_id})
        return result.deleted_count
    except Exception as exc:
        logger.error(f"删除文档台账失败：{exc}")
        return 0


def count_documents(tenant_id: str) -> int:
    """
    统计某租户的文档条数（用于判断是否需要从 Milvus 回填）

    :param tenant_id: 租户标识
    :return: 条数；失败返回 0
    """
    try:
        tool = get_document_mongo_tool()
        return tool.documents.count_documents({"tenant_id": tenant_id})
    except Exception as exc:
        logger.error(f"统计文档台账失败：{exc}")
        return 0


if __name__ == "__main__":
    # 自测：写入 → 列表 → 删除
    demo_tenant = "local"
    upsert_document("demo-doc-id", demo_tenant, "示例文档", "示例主体", 3, task_id="t-demo")
    for row in list_documents(demo_tenant, limit=5):
        print(row)
    print("删除条数：", delete_document("demo-doc-id", demo_tenant))
