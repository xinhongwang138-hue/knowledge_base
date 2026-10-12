"""
任务执行状态追踪工具

当前实现为**进程内内存追踪**：把每个 task 的每个节点的开始/结束时间记录在内存里，
供调试和后续接入持久化存储使用。

节点执行顺序为：
    add_running_task(task_id, node_name)  ->  节点执行  ->  add_done_task(task_id, node_name)

如果需要跨进程 / 重启后仍能查到历史，后续把 _TASK_STORE 换成
MongoDB（依赖里已有 pymongo）或 SQLite 即可，对外函数签名不变。
"""
import threading
import time
from datetime import datetime
from typing import Any, Dict

from app.core.logger import logger

# task_id -> {node_name: 状态记录}
# 用锁保护，避免多线程下并发写入丢数据
_TASK_STORE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_LOCK = threading.Lock()

# P1-2：模块级默认命名空间。不同业务服务（导入 / 查询）启动时调用
# set_default_namespace() 设置各自前缀，使同进程内两套 task_id / session_id
# 互不覆盖；未显式传 namespace 的调用自动带上该前缀。
_DEFAULT_NAMESPACE: str = ""


def set_default_namespace(namespace: str) -> None:
    """设置本进程内任务记录的默认命名空间（如 "import" / "query"）"""
    global _DEFAULT_NAMESPACE
    _DEFAULT_NAMESPACE = (namespace or "").strip()
    logger.info(f"task_utils 默认命名空间已设置为：{_DEFAULT_NAMESPACE or '(空)'}")


def _effective_ns(namespace: str = "") -> str:
    """显式传入的 namespace 优先，否则用模块默认命名空间"""
    return namespace or _DEFAULT_NAMESPACE


def _key_with_namespace(namespace: str, task_id: str) -> str:
    """带命名空间的存储键，避免不同业务（导入/查询）的 task_id 互相覆盖"""
    ns = _effective_ns(namespace)
    return f"{ns}:{task_id}" if ns else task_id


def _now_iso() -> str:
    """当前时间的 ISO 字符串"""
    return datetime.now().isoformat(timespec="milliseconds")


def _get_node_record(task_id: str, node_name: str, namespace: str = "") -> Dict[str, Any]:
    """取出（或初始化）某个 task 下某个节点的记录，加锁（P1-2：key 带命名空间）"""
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        task_nodes = _TASK_STORE.setdefault(key, {})
        return task_nodes.setdefault(
            node_name,
            {"task_id": task_id, "node_name": node_name, "status": "pending"},
        )


def add_running_task(task_id: str, node_name: str, namespace: str = "") -> Dict[str, Any]:
    """
    标记某个 task 的某个节点**开始执行**

    :param task_id: 任务唯一标识
    :param node_name: 节点标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: 该节点的最新状态记录
    """
    if not task_id:
        # task_id 为空时无法追踪，但不应因此中断业务流程，只告警
        logger.warning("add_running_task 收到空 task_id，跳过状态追踪")
        return {}

    record = _get_node_record(task_id, node_name, namespace)
    record.update(
        {
            "status": "running",
            "start_time": _now_iso(),
            "start_ts": time.time(),
        }
    )
    logger.info(f"▶ 任务开始：task_id={task_id}, 节点={node_name}")
    return record


def add_done_task(task_id: str, node_name: str, status: str = "success", namespace: str = "") -> Dict[str, Any]:
    """
    标记某个 task 的某个节点**执行结束**

    :param task_id: 任务唯一标识
    :param node_name: 节点标识
    :param status: 结束状态，默认 "success"，失败时传 "failed"
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: 该节点的最新状态记录
    """
    if not task_id:
        logger.warning("add_done_task 收到空 task_id，跳过状态追踪")
        return {}

    record = _get_node_record(task_id, node_name, namespace)
    record.update(
        {
            "status": status,
            "end_time": _now_iso(),
            "end_ts": time.time(),
        }
    )

    # 若有开始时间，算出耗时，便于定位慢节点
    if "start_ts" in record:
        cost_ms = int((record["end_ts"] - record["start_ts"]) * 1000)
        record["cost_ms"] = cost_ms
        if status == "success":
            logger.success(f"✔ 任务结束：task_id={task_id}, 节点={node_name}, 耗时={cost_ms}ms")
        else:
            logger.warning(f"✖ 任务失败：task_id={task_id}, 节点={node_name}, 耗时={cost_ms}ms")
    else:
        logger.info(f"✔ 任务结束：task_id={task_id}, 节点={node_name}")

    return record


def get_task_status(task_id: str, namespace: str = "") -> Dict[str, Dict[str, Any]]:
    """
    查询某个 task 下所有节点的状态快照

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: {node_name: 状态记录}；task 不存在时返回空 dict
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        return {name: dict(rec) for name, rec in _TASK_STORE.get(key, {}).items()}


def clear_task(task_id: str, namespace: str = "") -> None:
    """
    清空某个 task 的所有状态记录，避免内存无限增长

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        _TASK_STORE.pop(key, None)
    logger.debug(f"已清空任务状态：task_id={task_id}")

# ===================== 任务全局状态与列表查询（供 Web 服务轮询使用） =====================
# 上面是「节点级」追踪；下面是「任务级」追踪：一个 task 整体处于
# pending / processing / completed / failed 中的哪个阶段。
# 节点列表由 _TASK_STORE 派生，无需重复存一份。

# task_id -> 任务全局状态字符串
_TASK_GLOBAL_STATUS: Dict[str, str] = {}

# task_id -> 任务全局状态最后更新时间（Unix 秒），用于 TTL 清理
_TASK_GLOBAL_STATUS_TS: Dict[str, float] = {}

# 任务全局状态 / 结果的最大保留时长（秒），超时自动清理，防止长期运行内存无限增长
TASK_RECORD_TTL_SECONDS = 24 * 3600


def _key_with_namespace(namespace: str, task_id: str) -> str:
    """带命名空间的存储键，避免不同业务（导入/查询）的 task_id 互相覆盖"""
    ns = _effective_ns(namespace)
    return f"{ns}:{task_id}" if ns else task_id


def cleanup_task_records(ttl: float = TASK_RECORD_TTL_SECONDS) -> int:
    """
    清理超过 TTL 的任务全局状态与结果记录

    正常情况下任务结束即 clear_task / clear_task_result；
    本函数兜底清理异常中断遗留的旧记录。

    :param ttl: 保留时长阈值（秒）
    :return: 本次清理的任务数
    """
    now = time.time()
    with _LOCK:
        expired_status = [
            key for key, ts in _TASK_GLOBAL_STATUS_TS.items() if now - ts > ttl
        ]
        for key in expired_status:
            _TASK_GLOBAL_STATUS.pop(key, None)
            _TASK_GLOBAL_STATUS_TS.pop(key, None)
        expired_result = [
            key for key, ts in _TASK_RESULT_TS.items() if now - ts > ttl
        ]
        for key in expired_result:
            _TASK_RESULT.pop(key, None)
            _TASK_RESULT_TS.pop(key, None)
    if expired_status or expired_result:
        logger.info(
            f"清理过期任务记录：状态 {len(expired_status)} 条、结果 {len(expired_result)} 条"
        )
    return len(expired_status) + len(expired_result)


def update_task_status(task_id: str, status: str, namespace: str = "") -> str:
    """
    更新任务的**全局状态**（区别于节点状态）

    :param task_id: 任务唯一标识
    :param status: pending / processing / completed / failed
    :param namespace: 业务命名空间（如 "import" / "query"），避免不同业务 task_id 冲突
    :return: 更新后的状态
    """
    if not task_id:
        logger.warning("update_task_status 收到空 task_id，跳过状态更新")
        return status
    key = _key_with_namespace(namespace, task_id)
    cleanup_task_records()
    with _LOCK:
        _TASK_GLOBAL_STATUS[key] = status
        _TASK_GLOBAL_STATUS_TS[key] = time.time()
    logger.info(f"任务状态更新：task_id={task_id} -> {status}")
    return status


def get_global_status(task_id: str, namespace: str = "") -> str:
    """
    查询任务全局状态

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（如 "import" / "query"）
    :return: pending / processing / completed / failed；未知任务返回 pending
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        return _TASK_GLOBAL_STATUS.get(key, "pending")


def get_done_task_list(task_id: str, namespace: str = "") -> list:
    """
    查询任务已完成（success）的节点列表

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: 已完成节点名列表，按完成时间排序
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        records = _TASK_STORE.get(key, {})
        done = [
            (rec.get("end_ts", 0.0), name)
            for name, rec in records.items()
            if rec.get("status") == "success"
        ]
    return [name for _, name in sorted(done)]


def get_running_task_list(task_id: str, namespace: str = "") -> list:
    """
    查询任务正在执行（running）的节点列表

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: 运行中节点名列表
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        records = _TASK_STORE.get(key, {})
        running = [
            (rec.get("start_ts", 0.0), name)
            for name, rec in records.items()
            if rec.get("status") == "running"
        ]
    return [name for _, name in sorted(running)]


def get_failed_task_list(task_id: str, namespace: str = "") -> list:
    """
    查询任务执行失败（failed）的节点列表

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（默认取 set_default_namespace 设置的全局值）
    :return: 失败节点名列表
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        records = _TASK_STORE.get(key, {})
    return [name for name, rec in records.items() if rec.get("status") == "failed"]


# ===================== 任务状态常量与结果存储（供查询服务使用） =====================

# 任务全局状态的四种取值
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# task_id -> 任务执行结果（如最终答案），供非流式模式同步取用
_TASK_RESULT: Dict[str, Dict[str, Any]] = {}

# task_id -> 任务结果最后更新时间（Unix 秒），用于 TTL 清理
_TASK_RESULT_TS: Dict[str, float] = {}


def set_task_result(task_id: str, namespace: str = "", **results) -> Dict[str, Any]:
    """
    存储任务的执行结果（如最终答案），供非流式接口同步读取

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（如 "import" / "query"），避免不同业务 task_id 冲突
    :param **results: 键值对形式的结果，如 set_task_result(tid, answer="...")
    :return: 该任务当前全部结果
    """
    if not task_id:
        logger.warning("set_task_result 收到空 task_id，跳过结果存储")
        return {}
    key = _key_with_namespace(namespace, task_id)
    cleanup_task_records()
    with _LOCK:
        _TASK_RESULT.setdefault(key, {}).update(results)
        _TASK_RESULT_TS[key] = time.time()
        return dict(_TASK_RESULT[key])


def get_task_result(task_id: str, key: str, default: Any = None, namespace: str = "") -> Any:
    """
    读取任务执行结果的指定字段

    :param task_id: 任务唯一标识
    :param key: 结果字段名，如 "answer"
    :param default: 字段不存在时返回的默认值
    :param namespace: 业务命名空间（如 "import" / "query"）
    :return: 存储的结果值；任务不存在时返回 default
    """
    k = _key_with_namespace(namespace, task_id)
    with _LOCK:
        return _TASK_RESULT.get(k, {}).get(key, default)


def clear_task_result(task_id: str, namespace: str = "") -> None:
    """
    清空某任务的结果缓存

    :param task_id: 任务唯一标识
    :param namespace: 业务命名空间（如 "import" / "query"）
    """
    key = _key_with_namespace(namespace, task_id)
    with _LOCK:
        _TASK_RESULT.pop(key, None)
        _TASK_RESULT_TS.pop(key, None)
