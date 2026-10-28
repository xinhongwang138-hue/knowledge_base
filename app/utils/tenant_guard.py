"""租户解析与内部服务令牌校验（知识库租户隔离的唯一入口）。

设计目标：让"当前请求属于哪个租户"这个问题只有一个可信答案 —— 内部令牌。
电商后端作为唯一知道租户的地方，用 KB_INTERNAL_SECRET 签一个短时效令牌放进
X-KB-Internal 头；本模块验签后**只从令牌里取租户**，请求体/查询参数里客户端
自带的任何租户字段一律不参与决策。这样即使代理层出现拼接错误，智库侧也不会
因为"信了调用方传的 tenant"而跨租户读取。

两种运行模式（由环境变量切换，保证智库独立启动的老用法不变）：
    KB_REQUIRE_TENANT 未开启（默认）：独立本机模式。无令牌时落到 LOCAL_TENANT，
        沿用 access_guard 的来源校验，import.html / chat.html 照旧可用。
    KB_REQUIRE_TENANT=true（合并部署）：/upload、/query、/status、/history 等
        接口必须携带合法内部令牌，缺失或验签失败一律拒绝，**绝不退化为全库检索**。

令牌格式与电商侧 services/auth.py 保持同构（base64url(payload).base64url(hmac)），
但用的是独立的 KB_INTERNAL_SECRET：智库被攻破也伪造不出电商的用户登录令牌。
"""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict

from fastapi import HTTPException, Request

from app.utils.escape_milvus_string_utils import escape_milvus_string

# 独立本机模式下的租户名。用固定非空串而不是空串：Milvus 的 partition key 需要
# 一个真实取值，空串在部分版本上会被判为非法分区键。
LOCAL_TENANT = "local"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def tenant_required() -> bool:
    """是否处于"必须带内部令牌"的合并部署模式（每次读环境变量，避免导入期固化）"""
    return os.getenv("KB_REQUIRE_TENANT", "").strip().lower() in _TRUE_VALUES


def internal_secret() -> str:
    """内部令牌密钥；未配置返回空串"""
    return os.getenv("KB_INTERNAL_SECRET", "").strip()


def _base64url_decode(segment: str) -> bytes:
    """补齐 '=' 后解码 base64url 段"""
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _sign(segment: str, secret: str) -> str:
    """对 payload 段做 HMAC-SHA256 签名，返回 base64url 字符串"""
    digest = hmac.new(secret.encode("utf-8"), segment.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def parse_internal_token(token: str, secret: str) -> Dict[str, Any]:
    """
    校验内部令牌签名与有效期

    :param token: 形如 "<payload>.<signature>" 的令牌
    :param secret: KB_INTERNAL_SECRET
    :return: 校验通过的 payload
    :raises ValueError: 格式错误 / 签名不符 / 已过期
    """
    parts = token.split(".", 1)
    if len(parts) != 2:
        raise ValueError("内部令牌格式错误")
    encoded_payload, signature = parts

    # 先验签名再看内容，避免未受信载荷影响后续分支
    if not hmac.compare_digest(signature, _sign(encoded_payload, secret)):
        raise ValueError("内部令牌签名不符")

    try:
        payload = json.loads(_base64url_decode(encoded_payload))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("内部令牌载荷无法解析") from exc
    if not isinstance(payload, dict):
        raise ValueError("内部令牌载荷不是对象")

    expires_at = int(payload.get("exp", 0) or 0)
    if expires_at <= 0:
        raise ValueError("内部令牌缺少有效期")
    if expires_at < int(time.time()):
        raise ValueError("内部令牌已过期")
    return payload


def resolve_tenant(request: Request) -> str:
    """
    FastAPI 依赖：解析当前请求的租户，作为全链路唯一的租户来源

    :param request: 当前请求
    :return: 租户标识（合并部署下来自已验签的内部令牌，独立模式为 LOCAL_TENANT）
    :raises HTTPException: 合并部署模式下缺少/无效令牌（401/403）
    """
    token = (request.headers.get("x-kb-internal") or "").strip()
    secret = internal_secret()

    if not token:
        if tenant_required():
            raise HTTPException(
                status_code=401,
                detail="缺少 X-KB-Internal 内部服务令牌",
            )
        # 独立本机模式：没有租户概念，统一落到固定租户
        return LOCAL_TENANT

    if not secret:
        # 带了令牌却没配密钥，属于配置错误。合并模式下直接拒，
        # 独立模式下忽略令牌继续用本机租户，避免把老用法堵死。
        if tenant_required():
            raise HTTPException(status_code=503, detail="KB_INTERNAL_SECRET 未配置")
        return LOCAL_TENANT

    try:
        payload = parse_internal_token(token, secret)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=f"内部令牌无效：{exc}") from exc

    tenant = str(payload.get("tenant") or "").strip()
    if not tenant:
        raise HTTPException(status_code=401, detail="内部令牌缺少 tenant")
    return tenant


def scoped_state_patch(tenant_id: str) -> Dict[str, Any]:
    """
    给 LangGraph 初始状态打租户补丁

    :param tenant_id: resolve_tenant 的返回值
    :return: 可直接 update 进 state 的字典
    """
    return {"tenant_id": tenant_id}


def validate_tenant(tenant_id: str) -> str:
    """
    归一化并校验租户标识

    隔离的失败模式必须是"报错/查不到"，不能是"查全部"，所以所有读写 Milvus
    前的第一道关卡都是这个函数。

    :param tenant_id: 原始租户标识
    :return: 去掉首尾空白后的租户标识
    :raises ValueError: 租户为空
    """
    tenant = (tenant_id or "").strip()
    if not tenant:
        raise ValueError("租户标识为空，拒绝执行无租户范围的读写")
    return tenant


def require_tenant_filter(tenant_id: str) -> str:
    """
    生成 Milvus 租户过滤表达式片段，缺失租户即拒绝执行

    :param tenant_id: 当前租户
    :return: 形如 tenant_id == "ws_1" 的表达式
    :raises ValueError: 租户为空
    """
    tenant = validate_tenant(tenant_id)
    return f'tenant_id == "{escape_milvus_string(tenant)}"'


def combine_expr(*exprs: str) -> str:
    """
    合并多个过滤表达式，自动跳过空值

    :param exprs: 表达式片段
    :return: 用 and 连接的结果；全部为空返回 None
    """
    parts = [e for e in exprs if e and e.strip()]
    if not parts:
        return None
    return " and ".join(f"({p})" for p in parts)
