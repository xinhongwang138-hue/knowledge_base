"""
大模型客户端工具类

适配 OpenAI 兼容 API（千问/ 即梦 AI等），带全局缓存避免重复初始化。
"""
from typing import Optional

from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

from app.core.logger import logger
from app.lm.lm_config import lm_config, _DEFAULT_VL_MODEL, _DEFAULT_LLM_MODEL

# 全局缓存：键为(模型名, JSON 输出模式)，值为 ChatOpenAI 实例
_llm_client_cache = {}


def get_llm_client(
    model: Optional[str] = None,
    json_mode: bool = False,
    provider: str = "text",
) -> ChatOpenAI:
    """
    获取带全局缓存的 ChatOpenAI 客户端实例

    :param model: 模型名，优先级：入参 > lm_config.llm_model > 内置默认
    :param json_mode: 是否开启 JSON 标准输出模式
    :param provider: "text" 用主凭证（OPENAI_API_*）；"vision" 用视觉专属凭证
                      （VL_API_*，缺失时回退到主凭证）。用于「文本 SCNet / 视觉阿里云」混合部署
    :return: ChatOpenAI 实例
    :raise ValueError: 缺失 API 密钥 / 基础地址
    """
    if provider == "vision":
        # 视觉 provider：优先 VL_API_KEY，缺失回退主 key
        api_key = lm_config.vl_api_key or lm_config.api_key
        base_url = lm_config.vl_api_base or lm_config.base_url
        target_model = model or lm_config.lv_model or _DEFAULT_VL_MODEL
    else:
        api_key = lm_config.api_key
        base_url = lm_config.base_url
        target_model = model or lm_config.llm_model or _DEFAULT_LLM_MODEL

    cache_key = (provider, target_model, json_mode)

    if cache_key in _llm_client_cache:
        logger.debug(
            f"[LLM客户端] 缓存命中：provider={provider} 模型={target_model} JSON={json_mode}"
        )
        return _llm_client_cache[cache_key]

    if not api_key:
        raise ValueError(
            "[LLM客户端] 配置缺失：请在 .env 中配置 OPENAI_API_KEY（或视觉用 VL_API_KEY）\n"
            "  申请地址：https://bailian.console.aliyun.com/?tab=model#/api-key"
        )
    if not base_url:
        raise ValueError(
            "[LLM客户端] 配置缺失：请在 .env 中配置 OPENAI_API_BASE（或视觉用 VL_API_BASE）\n"
            "  参考：https://bailian.console.aliyun.com/?tab=doc#/doc"
        )
    logger.info(
        f"[LLM客户端] 初始化新实例：provider={provider} 模型={target_model} JSON={json_mode}"
    )

    extra_body = {"enable_thinking": False}
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM客户端] 已开启 JSON 输出模式")

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as exc:
        raise Exception(
            f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain 层）：{exc}"
        ) from exc

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}")
    return llm_client


if __name__ == "__main__":
    logger.info("===== 开始 LLM 客户端测试 =====")
    try:
        c1 = get_llm_client()
        logger.info("测试1：默认配置客户端创建成功")
        c2 = get_llm_client(model="DeepSeek-V4.1-Flash")
        logger.info("测试2：指定多模态模型创建成功")
        c3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"测试3：缓存命中，c2 is c3 = {c2 is c3}")
        get_llm_client(model="qwen-flash", json_mode=True)
        logger.info("测试4：JSON 输出模式创建成功")
    except Exception as exc:
        logger.error(f"LLM 客户端测试失败：{exc}")
    finally:
        logger.info("===== LLM 客户端测试结束 =====")
