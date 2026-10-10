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


def _now_iso() -> str:
    """当前时间的 ISO 字符串"""
    return datetime.now().isoformat(timespec="milliseconds")


def _get_node_record(task_id: str, node_name: str) -> Dict[str, Any]:
    """取出（或初始化）某个 task 下某个节点的记录，加锁"""
    with _LOCK:
        task_nodes = _TASK_STORE.setdefault(task_id, {})
        return task_nodes.setdefault(
            node_name,
            {"task_id": task_id, "node_name": node_name, "status": "pending"},
        )


def add_running_task(task_id: str, node_name: str) -> Dict[str, Any]:
    """
    标记某个 task 的某个节点**开始执行**

    :param task_id: 任务唯一标识
    :param node_name: 节点标识
    :return: 该节点的最新状态记录
    """
    if not task_id:
        # task_id 为空时无法追踪，但不应因此中断业务流程，只告警
        logger.warning("add_running_task 收到空 task_id，跳过状态追踪")
        return {}

    record = _get_node_record(task_id, node_name)
    record.update(
        {
            "status": "running",
            "start_time": _now_iso(),
            "start_ts": time.time(),
        }
    )
    logger.info(f"▶ 任务开始：task_id={task_id}, 节点={node_name}")
    return record


def add_done_task(task_id: str, node_name: str, status: str = "success") -> Dict[str, Any]:
    """
    标记某个 task 的某个节点**执行结束**

    :param task_id: 任务唯一标识
    :param node_name: 节点标识
    :param status: 结束状态，默认 "success"，失败时传 "failed"
    :return: 该节点的最新状态记录
    """
    if not task_id:
        logger.warning("add_done_task 收到空 task_id，跳过状态追踪")
        return {}

    record = _get_node_record(task_id, node_name)
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


def get_task_status(task_id: str) -> Dict[str, Dict[str, Any]]:
    """
    查询某个 task 下所有节点的状态快照

    :param task_id: 任务唯一标识
    :return: {node_name: 状态记录}；task 不存在时返回空 dict
    """
    with _LOCK:
        return {name: dict(rec) for name, rec in _TASK_STORE.get(task_id, {}).items()}


def clear_task(task_id: str) -> None:
    """
    清空某个 task 的所有状态记录，避免内存无限增长

    :param task_id: 任务唯一标识
    """
    with _LOCK:
        _TASK_STORE.pop(task_id, None)
    logger.debug(f"已清空任务状态：task_id={task_id}")

# ===================== 任务全局状态与列表查询（供 Web 服务轮询使用） =====================
# 上面是「节点级」追踪；下面是「任务级」追踪：一个 task 整体处于
# pending / processing / completed / failed 中的哪个阶段。
# 节点列表由 _TASK_STORE 派生，无需重复存一份。

# task_id -> 任务全局状态字符串
_TASK_GLOBAL_STATUS: Dict[str, str] = {}


def update_task_status(task_id: str, status: str) -> str:
    """
    更新任务的**全局状态**（区别于节点状态）

    :param task_id: 任务唯一标识
    :param status: pending / processing / completed / failed
    :return: 更新后的状态
    """
    if not task_id:
        logger.warning("update_task_status 收到空 task_id，跳过状态更新")
        return status
    with _LOCK:
        _TASK_GLOBAL_STATUS[task_id] = status
    logger.info(f"任务状态更新：task_id={task_id} -> {status}")
    return status


def get_global_status(task_id: str) -> str:
    """
    查询任务全局状态

    :param task_id: 任务唯一标识
    :return: pending / processing / completed / failed；未知任务返回 pending
    """
    with _LOCK:
        return _TASK_GLOBAL_STATUS.get(task_id, "pending")


def get_done_task_list(task_id: str) -> list:
    """
    查询任务已完成（success）的节点列表

    :param task_id: 任务唯一标识
    :return: 已完成节点名列表，按完成时间排序
    """
    with _LOCK:
        records = _TASK_STORE.get(task_id, {})
        done = [
            (rec.get("end_ts", 0.0), name)
            for name, rec in records.items()
            if rec.get("status") == "success"
        ]
    return [name for _, name in sorted(done)]


def get_running_task_list(task_id: str) -> list:
    """
    查询任务正在执行（running）的节点列表

    :param task_id: 任务唯一标识
    :return: 运行中节点名列表
    """
    with _LOCK:
        records = _TASK_STORE.get(task_id, {})
        running = [
            (rec.get("start_ts", 0.0), name)
            for name, rec in records.items()
            if rec.get("status") == "running"
        ]
    return [name for _, name in sorted(running)]


def get_failed_task_list(task_id: str) -> list:
    """
    查询任务执行失败（failed）的节点列表

    :param task_id: 任务唯一标识
    :return: 失败节点名列表
    """
    with _LOCK:
        records = _TASK_STORE.get(task_id, {})
    return [name for name, rec in records.items() if rec.get("status") == "failed"]


# ===================== 任务状态常量与结果存储（供查询服务使用） =====================

# 任务全局状态的四种取值
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# task_id -> 任务执行结果（如最终答案），供非流式模式同步取用
_TASK_RESULT: Dict[str, Dict[str, Any]] = {}


def set_task_result(task_id: str, **results) -> Dict[str, Any]:
    """
    存储任务的执行结果（如最终答案），供非流式接口同步读取

    :param task_id: 任务唯一标识
    :param **results: 键值对形式的结果，如 set_task_result(tid, answer="...")
    :return: 该任务当前全部结果
    """
    if not task_id:
        logger.warning("set_task_result 收到空 task_id，跳过结果存储")
        return {}
    with _LOCK:
        _TASK_RESULT.setdefault(task_id, {}).update(results)
        return dict(_TASK_RESULT[task_id])


def get_task_result(task_id: str, key: str, default: Any = None) -> Any:
    """
    读取任务执行结果的指定字段

    :param task_id: 任务唯一标识
    :param key: 结果字段名，如 "answer"
    :param default: 字段不存在时返回的默认值
    :return: 存储的结果值；任务不存在时返回 default
    """
    with _LOCK:
        return _TASK_RESULT.get(task_id, {}).get(key, default)


def clear_task_result(task_id: str) -> None:
    """
    清空某任务的结果缓存

    :param task_id: 任务唯一标识
    """
    with _LOCK:
        _TASK_RESULT.pop(task_id, None)
