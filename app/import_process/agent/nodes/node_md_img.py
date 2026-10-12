"""
    主要目标: 将md中的图片进行单独处理，图片转成对应的语义文本，方便后续切片搜索
    主要动作: 图片 -> 图片服务器 -> (上文)图片内容(下文) -> 视觉模型 -> 生成图片总结
             -> 替换md原有的图片显示 ![图片总结](图片的minio网络地址)
             -> state 更新 md_content / md_path -> 结束
    实现步骤:
          1. 进行任务和日志处理
          2. 核心参数校验 [校验md_path/md_content/返回images文件夹地址]
          3. 查找md中使用的图片和上下文 [返回 [(图片名, 图片地址, (上文, 下文))]
          4. 图片内容总结 [调用多模态模型，返回 {图片名: 总结}]
          5. 上传图片到minio，替换md中的本地地址和描述，返回新md_content
          6. 备份新的md内容，改为原名称 _new.md
          7. 更新 md_path / md_content 到 state
          8. 返回目标结果
"""
import base64
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.clients.minio_config import minio_config
from app.clients.minio_utils import get_minio_client
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_config import lm_config
from app.lm.llm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task, add_running_task

# MinIO 支持的图片格式（小写后缀，统一匹配）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 截取上下文的字符数
CONTEXT_CHARS = 100

# P2-7：图片摘要并发参数——并发数（I/O 密集，4 线程即可）与单张失败重试次数
_SUMMARY_MAX_WORKERS = 4
_SUMMARY_MAX_RETRIES = 2

# 图片识别 MIME（VLM 接口对图片类型敏感，png 写成 jpeg 也能识别，但不规范）
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def is_supported_image(filename: str) -> bool:
    """
    判断是否为支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回 True
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)
    """
    # 1. 任务和日志
    add_running_task(state['task_id'], 'node_md_img')
    try:
        # 2. 核心参数校验
        md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
        # 3. 查找 md 中使用的图片和上下文
        image_targets = step_2_scan_images(md_content, images_dir_obj)
        if not image_targets:
            logger.info("MD 中未发现可用图片，跳过图片处理")
            add_done_task(state['task_id'], 'node_md_img')
            return state
        # 4. 图篇内容总结
        image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
        # 全军覆没就别往下走了：step_4 会连 MinIO 客户端都连不上，白跑一趟
        if not image_summaries:
            logger.warning("所有图片摘要均生成失败，跳过上传与替换，MD 内容保持原样")
            add_done_task(state['task_id'], 'node_md_img', status="failed")
            return state
        # 5. 上传并替换
        new_md_content = step_4_upload_images_replace(
            image_summaries, image_targets, md_content, md_path_obj.stem
        )
        # 6. 备份为 {stem}_new.md
        new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
        # 7. 更新 state
        state['md_path'] = new_md_file_path_str
        state['md_content'] = new_md_content
    except Exception:
        add_done_task(state['task_id'], 'node_md_img', status="failed")
        raise
    add_done_task(state['task_id'], 'node_md_img')
    return state


@step_log("step_1 获取内容")
def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取和校验内容，返回 md 内容、md 路径、images 目录
    :param state: 流程状态
    :return: (md_content, md_path_obj, images_dir_obj)
    """
    # 1. md 路径校验
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md_path 参数错误，请检查输入参数！")
    md_file_obj = Path(md_file_path)
    if not md_file_obj.exists():
        raise FileNotFoundError(f"md 文件不存在：{md_file_path}")

    # 2. md 内容：state 里没有就从磁盘读
    md_content = state.get("md_content")
    if not md_content:
        md_content = md_file_obj.read_text(encoding="utf-8")
        state["md_content"] = md_content

    # 3. images 目录（与 md 同级）
    images_dir_obj = md_file_obj.parent / "images"

    return md_content, md_file_obj, images_dir_obj


@step_log("step_2 扫描图片")
def step_2_scan_images(
    md_content: str, images_dir_obj: Path
) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描 MD 中引用的本地图片，检查文件是否存在，并截取上下文（前后各100字符）

    :param md_content: Markdown 原文
    :param images_dir_obj: 图片所在目录
    :return: [(图片文件名, 图片完整路径, (上文, 下文))]
    """
    if not images_dir_obj.is_dir():
        logger.info(f"images 目录不存在，跳过扫描：{images_dir_obj}")
        return []

    image_targets = []
    for image_file in sorted(images_dir_obj.iterdir()):
        img_name = image_file.name

        # 过滤非图片
        if not is_supported_image(img_name):
            if image_file.is_file():
                logger.warning(f"跳过非图片文件：{img_name}")
            continue

        # 匹配 MD 图片语法 ![alt](路径含图片名)
        # re.escape 处理文件名中的特殊字符（括号、点等）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        items = list(pattern.finditer(md_content))
        if not items:
            logger.warning(f"图片 {img_name} 未在 MD 中引用，跳过")
            continue

        # 取第一处引用的位置，截取上下文
        start, end = items[0].span()
        pre_text = md_content[max(start - CONTEXT_CHARS, 0): start]
        post_text = md_content[end: min(end + CONTEXT_CHARS, len(md_content))]

        image_targets.append((img_name, str(image_file), (pre_text, post_text)))
        logger.debug(f"匹配到图片引用：{img_name}")

    logger.info(f"共匹配到 {len(image_targets)} 张被引用的图片")
    return image_targets


@step_log("step_3 生成图片摘要")
def step_3_image_summary(image_targets, stem: str) -> Dict[str, str]:
    """
    调用视觉模型总结图篇内容（P2-7：多图并发 + 单张失败重试）

    :param image_targets: [(文件名, 文件地址, (上文, 下文))]
    :param stem: 源文件名（提示词里要用）
    :return: {图片名: 总结}
    """
    summaries = {}
    # 限速队列放函数内：只限制本次调用内的整体速率，不是全局限流；
    # 并发线程共享同一个 deque，控制对模型侧的 QPS 冲击
    requests_limiter = deque()

    def _summarize_one(item) -> tuple:
        """单张图片摘要（内部重试），返回 (图片名, 摘要或 None)"""
        image_file, image_path, context = item
        # 访问限速（模型侧有 QPS 限制）
        apply_api_rate_limit(requests_limiter, max_requests=100)

        # P2-7：单张失败重试（视觉模型偶发限流/超时，重试一次可显著降低丢图率）
        for attempt in range(1, _SUMMARY_MAX_RETRIES + 1):
            try:
                path_obj = Path(image_path)
                prompt = load_prompt(
                    name="image_summary", root_folder=stem, image_content=context
                )

                # 图片转 base64（read_bytes 读二进制，不是 read_text）
                image_base64 = base64.b64encode(path_obj.read_bytes()).decode("utf-8")
                mime = _MIME_BY_EXT.get(path_obj.suffix.lower(), "image/jpeg")

                message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_base64}"},
                        },
                        {"type": "text", "text": prompt},
                    ]
                )

                # 视觉模型不需要 JSON 模式，走视觉专属 provider（VL_API_*，缺省回退主凭证）
                vl_client = get_llm_client(model=lm_config.lv_model, provider="vision")
                chain = vl_client | StrOutputParser()
                summary = chain.invoke([message]).strip()
                return image_file, summary
            except Exception as exc:
                if attempt < _SUMMARY_MAX_RETRIES:
                    logger.warning(
                        f"图片 {image_file} 摘要生成失败（第 {attempt} 次），将重试：{exc}"
                    )
                else:
                    # 重试用尽：不写入占位摘要——没有摘要的图片在 step_4 会被跳过上传，
                    # 避免用无意义的描述覆盖原有 alt 文本。
                    logger.exception(f"图片 {image_file} 摘要生成失败（重试 {_SUMMARY_MAX_RETRIES} 次）：{exc}")
        return image_file, None

    # P2-7：多图并发调用视觉模型（I/O 密集，线程池足够；池内异常已在 _summarize_one 吞掉）
    with ThreadPoolExecutor(max_workers=_SUMMARY_MAX_WORKERS) as pool:
        for image_file, summary in pool.map(_summarize_one, image_targets):
            if summary:
                summaries[image_file] = summary
                logger.info(f"图片 {image_file} 摘要：{summary}")

    return summaries


@step_log("step_4 上传并替换")
def step_4_upload_images_replace(
    image_summaries: Dict[str, str],
    image_targets,
    md_content: str,
    stem: str,
) -> str:
    """
    将图片上传到 MinIO，并替换 MD 中的本地路径与 alt 描述

    :param image_summaries: {图片名: 总结}
    :param image_targets: [(图片名, 图片路径, 上下文)]
    :param md_content: 原 md 内容
    :param stem: md 文件名（用于 MinIO 目录分层）
    :return: 替换后的 md_content
    """
    minio_client = get_minio_client()
    bucket = minio_config.bucket_name
    img_dir = minio_config.minio_img_dir  # 形如 upload-images（已无首尾斜杠）

    # 1. 清理该文档在 MinIO 中的旧图片，避免同名残留
    try:
        old_objects = list(
            minio_client.list_objects(bucket, prefix=f"{img_dir}/{stem}", recursive=True)
        )
        if old_objects:
            minio_client.remove_objects(
                bucket, delete_object_list=[DeleteObject(o.object_name) for o in old_objects]
            )
            logger.info(f"已清理 MinIO 中{stem} 的 {len(old_objects)} 个旧对象")
    except Exception as exc:
        logger.warning(f"清理旧对象失败（可忽略）：{exc}")

    # 2. 逐张上传，单张失败不影响整体
    images_urls = {}
    for image_file, image_path, _ in image_targets:
        if image_file not in image_summaries:
            continue  # 摘要失败的跳过，避免用占位描述覆盖
        try:
            minio_client.fput_object(
                bucket_name=bucket,
                object_name=f"{img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type=_MIME_BY_EXT.get(Path(image_file).suffix.lower(), "image/jpeg"),
            )
            # P1-5：桶默认私有。MD 落盘统一保存「公开直链形态」
            # （public_base_url/bucket/object_name）作为可重新签名的原始 URL，
            # 不固化 7 天过期的预签名串；对外返回时由查询链路
            # （node_answer_output / query_service）实时重签为预签名 URL。
            object_name = f"{img_dir}/{stem}/{image_file}"
            images_urls[image_file] = (
                f"{minio_config.public_base_url}/{bucket}/{object_name}"
            )
            logger.debug(f"上传成功：{image_file} -> {images_urls[image_file]}")
        except Exception as exc:
            logger.exception(f"上传图片失败：{image_file}，原因：{exc}")
            logger.debug("继续尝试上传下一张图片")

    if not images_urls:
        logger.warning("没有任何图片上传成功，MD 内容保持原样")
        return md_content

    # 3. 替换 md 内容：![任意描述](任意路径含图片名) -> ![摘要](在线URL)
    replaced = 0
    for image_file, url in images_urls.items():
        summary = image_summaries.get(image_file, image_file)
        # re.escape 必不可少：图片名带 () [] . 会直接让正则炸掉
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        # 用 lambda 而不是 f-string 替换：避免摘要里的 \1 \T 被当转义序列解析
        md_content, n = pattern.subn(lambda _: f"![{summary}]({url})", md_content)
        replaced += n

    logger.info(f"完成 MD 内容替换，共替换 {replaced} 处图片引用")
    return md_content


@step_log("step_5 备份md")
def step_5_backup_md_file(md_path_obj: Path, new_md_content: str) -> str:
    """
    把处理后的内容另存为 {stem}_new.md，返回新路径

    用 with_stem 会把后缀也换掉（xxx_new，无 .md），
    所以这里手动拼 parent / (stem + "_new" + suffix)
    """
    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    logger.info(f"已备份为：{new_md_path_obj.name}")
    return str(new_md_path_obj)


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时执行 MD 图片处理全流程"""
    from app.core.logger import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_path = PROJECT_ROOT / "output" / "测试带图文档" / "测试带图文档.md"

    if not test_md_path.exists():
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请先运行 node_pdf_to_md 生成 md，或手动放置测试 md 到 output 目录")
        raise SystemExit(1)

    # 检查 images 目录有没有图
    images_dir = test_md_path.parent / "images"
    if images_dir.is_dir():
        imgs = [p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        logger.info(f"images 目录发现 {len(imgs)} 张图片：{imgs}")
    else:
        logger.warning(f"images 目录不存在：{images_dir}（将跳过图片处理）")

    test_state = {
        "task_id": "test_task_123456",
        "md_path": str(test_md_path),
        "md_content": "",
    }
    logger.info("开始本地测试 - MD 图片处理全流程")
    result_state = node_md_img(test_state)
    logger.info(f"本地测试完成 - 新 md_path：{result_state['md_path']}")
    logger.info(f"内容预览：{result_state['md_content'][:200]}")