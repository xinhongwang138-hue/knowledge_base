"""
写接口访问控制（P1-4）。

设计目标：在不破坏现有前端（chat.html / import.html）的前提下，堵住"任意网页
通过浏览器对本机服务发起写操作"（CSRF / DNS rebinding）这条路。

两道闸门：
1. 来源校验（默认开启）：浏览器发起的写请求会带 Origin（或 Referer），
   其主机名必须是本机回环地址。恶意站点页面的 Origin 是外部域名，直接 403。
   非浏览器客户端（curl、脚本、容器内部调用）不带 Origin，不受影响。
2. 可选 API Key：在 .env 配置 WRITE_API_KEY 后，所有写请求必须带
   请求头 X-API-Key，否则 401。默认留空不启用，方便本机脚本免改造。

注意：本模块只做"来源 + 可选口令"校验，不做用户体系。服务端口已绑定
127.0.0.1，二者叠加后，外部主机无法访问，局域网内网页也无法借浏览器写入。
"""

import os
from urllib.parse import urlparse

from fastapi import HTTPException, Request

# 允许的来源主机名（含 IPv6 回环的各种写法）
_LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def _hostname_of(url: str) -> str:
    """
    从 Origin / Referer 里取出主机名。

    :param url: 形如 http://127.0.0.1:8011 的字符串
    :return: 小写主机名；解析失败返回空串
    """
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def verify_write_request(request: Request) -> None:
    """
    FastAPI 依赖：校验写请求来源，必要时校验 API Key。

    :param request: 当前请求
    :return: 校验通过返回 None
    :raises HTTPException: 来源不受信任（403）或缺少/错误的 Key（401）
    """
    # ---------- 闸门 1：来源必须是本机页面 ----------
    origin = request.headers.get("origin") or ""
    referer = request.headers.get("referer") or ""
    source = origin or referer
    if source:
        hostname = _hostname_of(source)
        # hostname 为空说明来源畸形，按不受信处理
        if not hostname or hostname not in _LOCAL_HOSTNAMES:
            raise HTTPException(
                status_code=403,
                detail="来源不被允许：写接口仅接受本机页面发起",
            )

    # ---------- 闸门 2：可选 API Key ----------
    # 每次读环境变量，避免模块导入时固化；留空即不启用
    expected_key = os.getenv("WRITE_API_KEY", "").strip()
    if expected_key:
        provided_key = (request.headers.get("x-api-key") or "").strip()
        if provided_key != expected_key:
            raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key")
