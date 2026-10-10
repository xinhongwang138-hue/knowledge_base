"""
提示词加载工具

所有提示词统一放在项目根的 prompts/ 目录下，便于集中管理。
文件命名：{name}.prompt，占位符用 Python str.format 语法。
"""
from app.core.logger import PROJECT_ROOT, logger

# 提示词存放目录：项目根 / prompts
PROMPT_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词并渲染占位符

    :param name: 提示词文件名（不带 .prompt 后缀，如 image_summary）
    :param **kwargs: 要渲染的变量（键名须与 .prompt 中的占位符完全一致）
    :return: 渲染后的最终提示词字符串
    :raise FileNotFoundError: 提示词文件不存在
    """
    prompt_path = PROMPT_DIR / f"{name}.prompt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"提示词文件不存在：{prompt_path}\n"
            f"请在 {PROMPT_DIR} 下创建 {name}.prompt"
        )

    raw_prompt = prompt_path.read_text(encoding="utf-8")

    if kwargs:
        return raw_prompt.format(**kwargs)
    return raw_prompt


def list_prompts() -> list[str]:
    """列出所有可用的提示词名（不含后缀）"""
    if not PROMPT_DIR.exists():
        return []
    return sorted(p.stem for p in PROMPT_DIR.glob("*.prompt"))


if __name__ == "__main__":
    logger.info("===== 开始提示词加载测试 =====")
    logger.info(f"提示词目录：{PROMPT_DIR}")
    logger.info(f"现有提示词：{list_prompts()}")

    try:
        rendered = load_prompt(
            name="image_summary",
            root_folder="hl3070使用说明书",
            image_content=("这是图片的上文内容", "这是图片的下文内容"),
        )
        logger.info("渲染后的提示词：\n" + rendered)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
    finally:
        logger.info("===== 提示词加载测试结束 =====")