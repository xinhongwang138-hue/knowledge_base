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

反馈数据结构（集合 chat_feedback，message_id 唯一）：
    {
        "message_id": 被评价的助手消息主键（与 chat_message._id 对应）
        "session_id": 会话唯一标识
        "tenant_id":  所属租户（合并部署下按租户隔离统计）
        "feedback":   up / down
        "comment":    用户补充的文字评论
        "question":   触发这条回答的提问（事后分析「哪类问题答不好」用）
        "ts":         时间戳（秒）
    }
    同一条回答反复评价只保留最后一次（upsert），不做追加插入，
    否则统计时一条回答会被计多次。
"""
import os
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
# 用户反馈集合：一条助手回答最多一条反馈记录（message_id 唯一）
FEEDBACK_COLLECTION_NAME = "chat_feedback"

# 反馈取值白名单：与前端按钮一一对应，避免脏数据进库
FEEDBACK_UP = "up"
FEEDBACK_DOWN = "down"
_FEEDBACK_VALUES = {FEEDBACK_UP, FEEDBACK_DOWN}

_FALSE_VALUES = {"0", "false", "no", "off"}


def history_enabled() -> bool:
    """
    是否启用对话历史 / 反馈读写（默认启用）

    离线评估（eval/eval.py）不需要历史，也不该要求本机有 MongoDB、
    更不该把评估产生的问答写进用户的真实会话记录里，
    因此提供 MONGO_HISTORY_ENABLED=false 开关；关闭后所有读写直接空转。

    :return: True 表示正常读写 Mongo；False 表示空转
    """
    return os.getenv("MONGO_HISTORY_ENABLED", "1").strip().lower() not in _FALSE_VALUES


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
        self.chat_feedback = self.db[FEEDBACK_COLLECTION_NAME]

        # 复合索引：session_id 升序 + ts 降序，
        # 适配「按会话查最新记录」的核心查询场景。
        # create_index 自带幂等性：索引已存在时不会重复创建。
        self.chat_message.create_index([("session_id", ASCENDING), ("ts", DESCENDING)])

        # 反馈以 message_id 为唯一键：同一条回答反复点👍👎只保留最新一次，
        # 不做追加插入，避免统计时同一条回答被计多次。
        try:
            self.chat_feedback.create_index([("message_id", ASCENDING)], unique=True)
            self.chat_feedback.create_index([("ts", DESCENDING)])
        except Exception as exc:
            # 历史脏数据可能已经产生重复 message_id，唯一索引建不上不该导致服务起不来，
            # 退化为普通索引继续可用（写入侧仍是 upsert，天然只有一条）。
            logger.warning(f"反馈集合唯一索引创建失败，降级为普通索引：{exc}")
            self.chat_feedback.create_index([("message_id", ASCENDING)])

        logger.info(
            f"成功连接到 MongoDB：{self.db_name}.{CHAT_COLLECTION_NAME}"
            f"（反馈集合 {FEEDBACK_COLLECTION_NAME}）"
        )

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
    sources: Optional[List[Dict[str, Any]]] = None,
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
    :param sources: 引用来源清单（引用溯源用，历史回看时还原「来源」区块）
    :return: 新增时返回 ObjectId 字符串，更新时返回传入的 message_id
    """
    if not history_enabled():
        return ""

    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        # 统一存成 list，避免 None 与 [] 两种空值给上层带来判断分支
        "item_names": list(item_names or []),
        "image_urls": list(image_urls or []),
        "sources": list(sources or []),
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
    if not history_enabled():
        return []

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
    清空指定会话的所有历史记录（连带清掉该会话的反馈）

    为什么连反馈一起删：反馈是挂在 message_id 上的，消息删了反馈还在，
    统计里就会混进「已经不存在的回答」，时间一长完全无法对账。

    :param session_id: 会话唯一标识
    :return: 实际删除的消息条数；失败返回 0
    """
    if not history_enabled():
        return 0

    try:
        tool = get_history_mongo_tool()
        result = tool.chat_message.delete_many({"session_id": session_id})
        # 反馈按会话清理：先删消息再删反馈，两者任一失败都不影响另一个的结果
        feedback_result = tool.chat_feedback.delete_many({"session_id": session_id})
        logger.info(
            f"已删除会话 {session_id} 的 {result.deleted_count} 条历史消息、"
            f"{feedback_result.deleted_count} 条反馈"
        )
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


def save_feedback(
    session_id: str,
    message_id: str,
    feedback: str,
    comment: str = "",
    tenant_id: str = "",
    question: str = "",
) -> Dict[str, Any]:
    """
    写入（或覆盖）一条用户反馈

    设计要点：
        · 以 message_id 为唯一键做 upsert —— 同一条回答反复点👍👎只留最新一次，
          不做追加插入，统计时不会把一条回答算多次；
        · 顺带存下当时的提问（question）与租户，便于后续做「哪类问题答不好」的分析，
          否则只凭 message_id 回头看历史还得再关联一次会话。

    :param session_id: 会话唯一标识
    :param message_id: 被评价的助手消息主键（Mongo _id 字符串）
    :param feedback: 反馈取值，up / down
    :param comment: 用户补充的文字评论（可空）
    :param tenant_id: 所属租户（合并部署下用于按租户隔离统计）
    :param question: 触发这条回答的用户提问（可空）
    :return: 落库后的文档摘要；参数非法或未启用历史时返回空字典
    """
    if feedback not in _FEEDBACK_VALUES:
        logger.warning(f"非法反馈取值：{feedback!r}，仅接受 {sorted(_FEEDBACK_VALUES)}")
        return {}
    if not history_enabled():
        return {}

    ts = datetime.now().timestamp()
    document = {
        "message_id": message_id,
        "session_id": session_id,
        "tenant_id": tenant_id or "",
        "feedback": feedback,
        "comment": (comment or "").strip(),
        "question": (question or "").strip(),
        "ts": ts,
    }

    try:
        tool = get_history_mongo_tool()
        tool.chat_feedback.update_one(
            {"message_id": message_id},
            {"$set": document},
            upsert=True,
        )
        logger.info(
            f"[{session_id}] 已记录反馈 {feedback}（message_id={message_id}，"
            f"评论长度={len(document['comment'])}）"
        )
        return document
    except Exception as exc:
        logger.error(f"写入反馈失败：{exc}")
        return {}


def get_message_by_id(
    message_id: str, session_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    按主键读取一条会话消息（可附带 session_id 校验归属）

    用途：写反馈前确认「这条回答确实属于这个会话」，避免前端传错 id
    把反馈挂到别人的消息上。

    :param message_id: 消息主键字符串
    :param session_id: 传入时要求消息属于该会话
    :return: 消息文档；不存在 / 不属于该会话返回 None
    """
    if not history_enabled() or not message_id:
        return None
    try:
        ObjectId(message_id)
    except Exception:
        return None

    query: Dict[str, Any] = {"_id": ObjectId(message_id)}
    if session_id:
        query["session_id"] = session_id
    try:
        tool = get_history_mongo_tool()
        return tool.chat_message.find_one(query)
    except Exception as exc:
        logger.error(f"查询消息失败：{exc}")
        return None


def get_question_before(session_id: str, ts: float) -> str:
    """
    取某条助手回答之前最近的一条用户提问

    :param session_id: 会话唯一标识
    :param ts: 助手回答的时间戳（秒）
    :return: 提问文本；找不到返回空串
    """
    if not history_enabled():
        return ""
    try:
        tool = get_history_mongo_tool()
        doc = tool.chat_message.find_one(
            {"session_id": session_id, "role": "user", "ts": {"$lt": ts}},
            sort=[("ts", DESCENDING)],
        )
        return (doc or {}).get("text", "")
    except Exception as exc:
        logger.error(f"查询上一轮提问失败：{exc}")
        return ""


def get_feedback_map(message_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    批量取反馈，返回 {message_id: {feedback, comment, ts}} 映射

    :param message_ids: 消息主键列表
    :return: 反馈映射；无反馈的 message_id 不在结果中
    """
    if not history_enabled() or not message_ids:
        return {}
    try:
        tool = get_history_mongo_tool()
        cursor = tool.chat_feedback.find({"message_id": {"$in": list(message_ids)}})
        return {
            doc["message_id"]: {
                "feedback": doc.get("feedback", ""),
                "comment": doc.get("comment", ""),
                "ts": doc.get("ts"),
            }
            for doc in cursor
        }
    except Exception as exc:
        logger.error(f"批量查询反馈失败：{exc}")
        return {}


def get_feedback_stats(days: int = 7, limit: int = 20, tenant_id: str = "") -> Dict[str, Any]:
    """
    反馈统计（供后续调参 / 分析使用）

    :param days: 统计最近多少天
    :param limit: 返回最近的多少条差评明细
    :param tenant_id: 传入时只统计该租户
    :return: {"up": n, "down": n, "total": n, "good_rate": 0.0~1.0,
              "recent_down": [{message_id, session_id, question, comment, ts}, ...]}
    """
    empty = {"up": 0, "down": 0, "total": 0, "good_rate": None, "recent_down": []}
    if not history_enabled():
        return empty

    since = datetime.now().timestamp() - max(days, 1) * 86400
    query: Dict[str, Any] = {"ts": {"$gte": since}}
    if tenant_id:
        query["tenant_id"] = tenant_id

    try:
        tool = get_history_mongo_tool()
        up = tool.chat_feedback.count_documents({**query, "feedback": FEEDBACK_UP})
        down = tool.chat_feedback.count_documents({**query, "feedback": FEEDBACK_DOWN})
        total = up + down
        recent_down = [
            {
                "message_id": doc.get("message_id", ""),
                "session_id": doc.get("session_id", ""),
                "question": doc.get("question", ""),
                "comment": doc.get("comment", ""),
                "ts": doc.get("ts"),
            }
            for doc in tool.chat_feedback.find(
                {**query, "feedback": FEEDBACK_DOWN}, sort=[("ts", DESCENDING)], limit=limit
            )
        ]
        return {
            "up": up,
            "down": down,
            "total": total,
            # 无反馈时返回 None 而不是 0.0：0.0 会被误读成「全差评」
            "good_rate": round(up / total, 4) if total else None,
            "recent_down": recent_down,
        }
    except Exception as exc:
        logger.error(f"反馈统计失败：{exc}")
        return empty


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
