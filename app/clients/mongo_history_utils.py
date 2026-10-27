"""
MongoDB 历史对话记录读写工具（基于原生 PyMongo 实现）

为什么用 MongoDB 而不是 LangChain 的 Checkpointer：
    - 更灵活：可按任意条件组合查询（按会话、按时间、按商品），Checkpointer 的
      存取接口被 LangGraph 约定死，不便自定义
    - 可控：文档结构由我们自己定义，能塞进 rewritten_query / item_names /
      image_urls 这些业务字段
    - 契合：对话记录本身就是「结构简单、数据量大、查询简单」的 JSON 文档，
      正是 MongoDB 的舒适区

对话数据结构：
    {
        "session_id":     会话唯一标识
        "role":           角色 user / assistant
        "text":           对话文本（用户提问或助手回答）
        "rewritten_query": 改写后的问题（检索增强时由主体识别节点产出）
        "item_names":     关联商品名称列表
        "image_urls":     关联图片URL 列表
        "ts":             时间戳（秒），排序与筛选维度
    }

索引设计：
    复合索引 (session_id 升序, ts 降序) —— 精确适配「按会话取最近 N 条并按时间正序」
    这个核心查询，避免先按 session_id 过滤后再对全量结果做内存排序。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.logger import PROJECT_ROOT, logger

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")

# 集合名：相当于关系型数据库里的「表」
CHAT_COLLECTION_NAME = "chat_message"


class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类（基于原生 PyMongo）

    核心功能：封装 MongoDB 的连接、集合初始化、索引创建，
    为上层提供统一的数据库操作入口。
    """

    def __init__(self):
        """
        类初始化：完成 MongoDB 的连接、数据库/集合获取、索引创建。

        初始化失败会抛出异常，让调用方感知连接问题（而不是拿到一个半初始化的实例）。
        """
        import os

        self.mongo_url = os.getenv("MONGO_URL")
        self.db_name = os.getenv("MONGO_DB_NAME")

        if not self.mongo_url or not self.db_name:
            raise ValueError(
                "MongoDB 配置缺失：请在 .env 中配置 MONGO_URL 与 MONGO_DB_NAME"
            )

        # 创建客户端并建立连接
        self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.db_name]
        # MongoDB 不需要显式建库/建集合，写入时自动创建
        self.chat_message = self.db[CHAT_COLLECTION_NAME]

        # 复合索引：session_id 升序 + ts 降序，
        # 适配「按会话查最新记录」的核心查询场景。
        # create_index 自带幂等性：索引已存在时不会重复创建。
        self.chat_message.create_index([("session_id", ASCENDING), ("ts", DESCENDING)])

        logger.info(f"成功连接到 MongoDB：{self.db_name}.{CHAT_COLLECTION_NAME}")

    def ping(self) -> bool:
        """
        探活：确认数据库真正可读写（而不只是构造了对象）

        :return: True 表示连接正常
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception as exc:
            logger.error(f"MongoDB 探活失败：{exc}")
            return False


# 全局单例：避免重复建立数据库连接
_history_mongo_tool: Optional[HistoryMongoTool] = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取 HistoryMongoTool 的单例（懒加载）

    模块加载时不立即连接 —— MongoDB 未启动时不应让整个程序 import 失败，
    推迟到第一次真正调用历史记录功能时才连接。

    :return: HistoryMongoTool 单例
    """
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    写入/更新单条会话记录到 MongoDB

    支持两种模式：无 message_id 时新增，有 message_id 时更新已有记录。

    :param session_id: 会话唯一标识
    :param role: 消息角色 user / assistant
    :param text: 对话核心内容
    :param rewritten_query: 改写后的查询（默认空串）
    :param item_names: 关联的商品名称列表
    :param image_urls: 关联的图片URL 列表
    :param message_id: 记录主键；无值则新增，有值则更新
    :return: 新增时返回 ObjectId 字符串，更新时返回传入的 message_id
    """
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        # 统一存成 list，避免 None 与 [] 两种空值给上层带来判断分支
        "item_names": list(item_names or []),
        "image_urls": list(image_urls or []),
        "ts": ts,
    }

    tool = get_history_mongo_tool()
    if message_id:
        tool.chat_message.update_one({"_id": ObjectId(message_id)}, {"$set": document})
        return message_id

    result = tool.chat_message.insert_one(document)
    return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近 N 条对话记录，按时间**正序**返回（可直接喂给 LLM 作上下文）

    :param session_id: 会话唯一标识
    :param limit: 条数限制
    :return: 记录列表（原始字典）；查询失败返回空列表
    """
    try:
        tool = get_history_mongo_tool()
        cursor = (
            tool.chat_message.find({"session_id": session_id})
            # 先按 ts 倒序取最近 limit 条（走 (session_id, ts) 复合索引）
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        msgs = list(cursor)
        # 取出后再反转为正序，让最新的一条在最后（符合 LLM 阅读习惯）
        msgs.reverse()
        return msgs
    except Exception as exc:
        logger.error(f"查询历史消息失败：{exc}")
        return []


def clear_history(session_id: str) -> int:
    """
    清空指定会话的所有历史记录

    :param session_id: 会话唯一标识
    :return: 实际删除条数；失败返回 0
    """
    try:
        tool = get_history_mongo_tool()
        result = tool.chat_message.delete_many({"session_id": session_id})
        logger.info(f"已删除会话 {session_id} 的 {result.deleted_count} 条历史消息")
        return result.deleted_count
    except Exception as exc:
        logger.error(f"清空历史失败：{exc}")
        return 0


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史记录的关联商品名称

    典型场景：用户提问时还不知道商品，主体识别节点后续才确定型号，
    此时回填 item_names。

    :param ids: 要更新的记录主键 ID 字符串列表
    :param item_names: 新的商品名称列表
    :return: 实际修改条数；失败返回 0
    """
    try:
        tool = get_history_mongo_tool()
        object_ids = [ObjectId(i) for i in ids]
        result = tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": list(item_names or [])}},
        )
        return result.modified_count
    except Exception as exc:
        logger.error(f"批量更新 item_names 失败：{exc}")
        return 0


if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "你好 (Hybrid)")
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo 的助手。")
    save_chat_message(sid, "user", "这个万用表怎么换电池？", item_names=["混合万用表"])

    print("--- 查询最近 5 条 ---")
    for m in get_recent_messages(sid, limit=5):
        print(f"  [{m['ts']:.0f}] {m['role']:9} {m['text']}  item_names={m['item_names']}")
    print(f"共 {len(get_recent_messages(sid, limit=5))} 条")
    print("MongoDB 历史对话工具自测结束")
