"""
阿里云百炼 MCP（联网搜索）配置

MCP (Model Context Protocol) 是 Anthropic 提出的协议，让大模型以统一方式调用外部工具。
百炼把「联网搜索」做成了 MCP 服务，本项目通过 openai-agents SDK 调用它。

三种传输方式对比（笔记 9.4.2 的选型说明）：
    - stdio            本机子进程通信，适合本机工具/桌面插件
    - SSE              HTTP 下行流，早期 Web 方案，上下行分离、代理兼容性麻烦
    - Streamable HTTP  统一可流式传输，云上生产首选
本项目用 Streamable HTTP —— 笔记明确指出：新开通的 bailian_web_search 默认是
streamablehttp 而非 sse。

【当前状态】密钥未配置（空字符串）。此时节点会优雅降级为「联网搜索无结果」，
不会阻断检索主链路。开通方式见笔记 9.4.1，拿到地址后填入 .env 即可。
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.core.logger import PROJECT_ROOT

# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class McpConfig:
    """百炼 MCP 联网搜索配置"""

    # MCP 服务地址（Streamable HTTP）
    mcp_base_url: str = ""
    # 阿里云/百炼 API 密钥
    api_key: str = ""

    def __post_init__(self):
        if not self.mcp_base_url:
            # 两个地址都读：优先 streamable，回退 sse
            self.mcp_base_url = (
                os.getenv("MCP_DASHSCOPE_BASE_URL_STREAMABLE")
                or os.getenv("MCP_DASHSCOPE_BASE_URL")
                or ""
            )
        if not self.api_key:
            #复用项目已有的阿里云凭证（VL_API_KEY），不额外要求配置
            self.api_key = os.getenv("VL_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

        self.mcp_base_url = self.mcp_base_url.strip().strip('"').strip("'")
        self.api_key = self.api_key.strip().strip('"').strip("'")

    @property
    def is_ready(self) -> bool:
        """是否具备调用条件（地址 + 密钥都齐了）"""
        return bool(self.mcp_base_url and self.api_key)


mcp_config = McpConfig()


if __name__ == "__main__":
    print(f"mcp_base_url = {mcp_config.mcp_base_url or '(未配置)'}")
    print(f"api_key     = {'已配置(%d字符)' % len(mcp_config.api_key) if mcp_config.api_key else '(未配置)'}")
    print(f"is_ready    = {mcp_config.is_ready}")
