"""
日志模块：基于 loguru 的统一日志出口

提供两样东西：
1. logger    —— 全局日志单例，直接用logger.info(...) 等
2. node_log  —— 装饰器工厂，给 LangGraph 节点自动加上「进入/耗时/异常」日志

配置从项目根目录的 .env 读取，全部缺省安全：
    LOG_CONSOLE_ENABLE   控制台开关      默认 True
    LOG_CONSOLE_LEVEL    控制台级别      默认 INFO
    LOG_FILE_ENABLE      文件日志开关    默认 True
    LOG_FILE_LEVEL       文件级别        默认 DEBUG
    LOG_FILE_RETENTION   文件保留策略    默认 "7 days"
"""
import os
import sys
import functools
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from loguru import logger as _loguru_logger

# 项目根目录 = 本文件上溯 3 级（app/core/logger.py -> 根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 先加载 .env，让下面读配置能拿到值
load_dotenv(PROJECT_ROOT / ".env")

# 日志格式：时间 | 级别 | 所在文件:行号 | 消息
_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def _env_flag(key: str, default: bool) -> bool:
    """读取布尔型环境变量，容错 True/False/1/0/yes/no"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().strip('"').strip("'").lower() in ("true", "1", "yes", "on")


def _env_level(key: str, default: str) -> str:
    """读取日志级别名，非法值回退到默认"""
    raw = os.getenv(key)
    if raw is None:
        return default
    raw = raw.strip().strip('"').strip("'").upper()
    allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    return raw if raw in allowed else default


def _setup_logger():
    """按配置装配 loguru，返回配置好的 logger 单例"""
    logger_ = _loguru_logger
    # 先移除默认 handler，避免重复输出
    logger_.remove()

    console_on = _env_flag("LOG_CONSOLE_ENABLE", True)
    console_level = _env_level("LOG_CONSOLE_LEVEL", "INFO")
    file_on = _env_flag("LOG_FILE_ENABLE", True)
    file_level = _env_level("LOG_FILE_LEVEL", "DEBUG")
    retention = os.getenv("LOG_FILE_RETENTION", "7 days")

    if console_on:
        logger_.add(
            sys.stderr,
            level=console_level,
            format=_FMT,
            colorize=True,
            enqueue=False,
        )

    if file_on:
        log_dir = PROJECT_ROOT / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            logger_.add(
                log_dir / "app_{time:YYYY-MM-DD}.log",
                level=file_level,
                format=_FMT,
                encoding="utf-8",
                rotation="00:00",
                retention=retention,
                enqueue=False,
            )
        except Exception as exc:  # 日志文件写不了不该拖垮业务，降级为仅控制台
            # 此时控制台 sink 可能还没建好，直接写 stderr 保证可见
            print(f"[logger] 文件日志初始化失败，仅启用控制台日志：{exc}", file=sys.stderr)

    if not console_on and not file_on:
        # 两个开关都关时给个最简handler，保证 logger 至少有sink 可用
        logger_.add(sys.stderr, level="CRITICAL")

    return logger_


# 全局日志单例
logger = _setup_logger()


def node_log(node_name: str) -> Callable:
    """
    节点日志装饰器工厂：给 LangGraph 节点包一层「进入 + 耗时 + 异常」日志

    用法：
        @node_log("node_md_img")
        def node_md_img(state): ...

    :param node_name: 节点标识，会出现在日志里便于定位
    :return: 装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"▶[{node_name}] 开始执行")
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logger.exception(f"✖[{node_name}] 执行失败：{type(exc).__name__}: {exc}")
                raise
            logger.success(f"✔[{node_name}] 执行完成")
            return result

        return wrapper

    return decorator


def step_log(step_name: str) -> Callable:
    """
    步骤日志装饰器工厂：标记长流程中的一个可观测步骤

    与 node_log 的区别：
        node_log  用于 LangGraph 节点（进入/完成/异常/耗时）
        step_log  用于节点内部的子步骤（调用外部 API、轮询、下载、解压等），
                  只在真正执行到该步骤时打一条，不做耗时统计

    用途：像 MinerU 解析这种「上传→轮询→下载→解压」的多阶段流程，
    出问题时能立刻看出卡在哪一步，而不用翻一堆堆栈。

    用法：
        @step_log("轮询解析结果")
        def poll(...): ...

    :param step_name: 步骤名称，会出现在日志里
    :return: 装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"  ├─ [{step_name}] 开始")
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logger.exception(f"  ├─ [{step_name}] 失败：{type(exc).__name__}: {exc}")
                raise
            logger.info(f"  ├─ [{step_name}] 完成")
            return result

        return wrapper

    return decorator