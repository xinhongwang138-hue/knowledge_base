"""
大模型（LLM / VL）配置

只负责「配置从哪来、怎么校验」，不含业务逻辑。
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 用绝对路径加载 .env，避免依赖运行时的工作目录
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")

# 缺省模型名
_DEFAULT_LLM_MODEL = "qwen-flash"
_DEFAULT_VL_MODEL = "qwen3-vl-flash"
# 视觉模型的默认 endpoint（阿里云百炼 OpenAI 兼容地址）
_DEFAULT_VL_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMConfig:
    """大模型与视觉模型配置（均可通过 .env 覆盖）

    支持「双 provider」：
      - 文本/默认模型走 OPENAI_API_BASE + OPENAI_API_KEY
      - 视觉模型（图片摘要）走 VL_API_BASE + VL_API_KEY，
        不配则回退到文本那套凭证（即单 provider 模式，兼容旧配置）
    """

    base_url: str = field(default="")
    api_key: str = field(default="")
    # 视觉模型（多模态，用于图片摘要）
    lv_model: str = field(default="")
    # 文本模型
    llm_model: str = field(default="")
    # 温度，低温度保证输出确定性
    llm_temperature: float = 0.1

    # —— 视觉专属 provider（可选，用于「文本 SCNet / 视觉阿里云」混合部署）——
    vl_api_base: str = field(default="")
    vl_api_key: str = field(default="")

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_API_BASE", "")
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.lv_model:
            self.lv_model = os.getenv("VL_MODEL", "") or _DEFAULT_VL_MODEL
        if not self.llm_model:
            self.llm_model = os.getenv("LLM_DEFAULT_MODEL", "") or _DEFAULT_LLM_MODEL

        # 视觉 provider：缺省回退到阿里云百炼；密钥缺省回退到主密钥
        if not self.vl_api_base:
            self.vl_api_base = os.getenv("VL_API_BASE", "") or _DEFAULT_VL_API_BASE
        if not self.vl_api_key:
            self.vl_api_key = os.getenv("VL_API_KEY", "") or self.api_key

        temp = os.getenv("LLM_DEFAULT_TEMPERATURE")
        if temp and not self.llm_temperature:
            try:
                self.llm_temperature = float(temp)
            except ValueError:
                pass  # 配错了就用默认值 0.1


# 全局单例
lm_config = LLMConfig()


if __name__ == "__main__":
    print(f"base_url       = {lm_config.base_url or '(未配置)'}")
    print(f"api_key        = {lm_config.api_key[:8] + '...' if lm_config.api_key else '(未配置)'}")
    print(f"lv_model       = {lm_config.lv_model}")
    print(f"llm_model      = {lm_config.llm_model}")
    print(f"llm_temperature= {lm_config.llm_temperature}")
