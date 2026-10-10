"""
API 访问限速工具

用令牌桶算法的简化版：记录最近 N 次访问时间戳，
若在窗口期内已达上限，则休眠到可以继续访问。
"""
import time
from collections import deque
from typing import Deque

from app.core.logger import logger


def apply_api_rate_limit(
    request_times: Deque[float],
    max_requests: int = 10,
    period_seconds: float = 60.0,
) -> None:
    """
    API 访问限速（滑动窗口版令牌桶）

    典型用法：在循环里调模型前先调用一次，
    就会自动保证「period_seconds 内不超过 max_requests 次」

    :param request_times: 记录访问时间戳的队列，**由调用方创建并传入**
                          （这样每次函数调用各自独立限速，不是全局限流）
    :param max_requests: 窗口期内的最大访问次数
    :param period_seconds: 窗口周期（秒）
    """
    now = time.monotonic()

    # 1. 先剔除窗口期外的旧记录
    while request_times and now - request_times[0] > period_seconds:
        request_times.popleft()

    # 2. 已达上限则休眠到最早那条记录过期
    if len(request_times) >= max_requests:
        sleep_for = period_seconds - (now - request_times[0])
        if sleep_for > 0:
            logger.debug(
                f"[限速] 已达 {max_requests} 次/{period_seconds}s 上限，"
                f"休眠 {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)
            # 休眠后重新清理一次
            now = time.monotonic()
            while request_times and now - request_times[0] > period_seconds:
                request_times.popleft()

    # 3. 记录本次访问
    request_times.append(now)


if __name__ == "__main__":
    logger.info("===== 开始限速工具测试（max=3次/2秒）=====")
    q: Deque[float] = deque()
    start = time.monotonic()
    for i in range(6):
        apply_api_rate_limit(q, max_requests=3, period_seconds=2.0)
        logger.info(f"第 {i + 1} 次调用通过，累计 {i + 1}")
    cost = time.monotonic() - start
    logger.info(f"6 次调用总耗时 {cost:.2f}s（应约 2s，因第 4 次起触发限速）")
    logger.info("===== 限速工具测试结束 =====")