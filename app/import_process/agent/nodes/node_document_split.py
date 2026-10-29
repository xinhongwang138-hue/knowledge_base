"""
    主要目标: 将上游（node_entry / node_pdf_to_md / node_md_img）产出的 Markdown 长文本
             切分成小而语义完整的文本块(Chunk)，供后续主体识别、向量化与检索使用
    主要动作: 读取并清洗 md 内容 -> 按 Markdown 标题做语义初切（代码块内不误切）
             -> 超长块再递归精细化切割 -> 组装带元数据的 chunks
             -> 备份 backup_chunks.json + 更新 state["chunks"] -> 结束
    实现步骤:
          1. 进行任务和日志处理
          2. 获取并清洗内容 [统一换行符 \r\n / \r -> \n，取 file_title 兜底]
          3. 按标题语义初切 [正则匹配 #~######，跳过代码块，无标题兜底为「无主题」]
          4. 超长块递归精细化切割 [RecursiveCharacterTextSplitter 按 段落/换行/句子/空格 优先级]
          5. 组装标准 Chunk 结构 [title / content / file_title / parent_title / part]
          6. 备份到 backup_chunks.json + 更新 state["chunks"] 并返回
"""
import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState
from app.utils.html_table_utils import html_tables_to_markdown
from app.utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
# 单个文本块最大长度（控制不超过模型上下文）。
# 取值依据：一份中文说明书/手册里，一个 `##` 小节通常在 500~2000 字之间。
# 切到 400 字（约 5~8 条要点）时，问「某个具体细节」能命中一个语义集中的切片；
# 切到 200 会把一条完整警告拆成两半（问题里的条件和结论落到不同切片），
# 而完全不切（最早的实现）会让一整节 1700 字压成一个切片，向量被稀释，
# 问细节时排名靠后检不到——实测 23 条评估里有 7 条因此召回为 0。
CHUNK_SIZE = 400
# 块之间重叠长度（保证语义不丢失，关键信息不落在切口上）
CHUNK_OVERLAP = 60
# 递归切分优先级：段落 -> 换行 -> 中文句末标点 -> 空格。
# 越靠前切出的块语义越完整，最后才退到逐字符兜底
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " "]


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    # 1. 进行任务和日志处理
    add_running_task(state['task_id'], 'node_document_split')
    # 2. 进行 state 中数据清洗（md_content / file_title 兜底）
    md_content, file_title = step_1_get_content(state)
    # 3. 按标题语义初切
    # 返回结构：[{content: 标题内容, title: 标题, file_title: 文件名}, ...]
    sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)
    logger.info(f"标题初切完成：{title_count} 个标题块，共 {lines_count} 行")
    # 4. 进行语义内递归切割
    # 返回结构：[{content, title, file_title, parent_title, part}, ...]
    final_chunks = step_3_refine_chunks(sections)
    # 5. 数据备份和修改 state chunks
    state['chunks'] = final_chunks
    step_4_backup_chunks(final_chunks, state)
    add_done_task(state['task_id'], 'node_document_split')
    return state


@step_log("step_1_get_content")
def step_1_get_content(state) -> Tuple[str, str]:
    """
    数据清洗：处理 md_content 中不同系统的换行分隔符，统一处理，
    并获取 file_title 用于整篇内容的标题兜底。
    :param state: 流程状态字典，含 md_content / file_title
    :return: (清洗后的 md_content, file_title)
    """
    # 1. 获取 md_content 内容
    md_content = state.get('md_content')
    if not md_content:
        logger.error("没有输入内容，请检查上游节点是否正确写入 md_content")
        raise RuntimeError("没有输入内容，请检查输入内容是否正确!")
    # 2. 清洗数据：统一换行符号
    #    windows \r\n / 老 mac \r 都归一为 \n，保证跨平台一致
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    # 2.1 把 MinerU 输出的 HTML 表格摊平成 Markdown 表格。
    #     原始 HTML 表格是一整块无换行、无标点的长文本，切分器根本切不动，
    #     CHUNK_SIZE 对表格完全失效（实测一张表压成 1386 字的单片）。
    md_content = html_tables_to_markdown(md_content)
    file_title = state.get("file_title", "default_file")
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title) -> List[Dict]:
    """
    语义切分：根据 Markdown 标题(#~######)进行内容切割。
    自动跳过代码块内的 # 行（避免误切代码注释），保证每个块语义完整。
    :param md_content: 清洗后的 Markdown 全文
    :param file_title: 文件名（存入每块元数据，便于溯源）
    :return: [{content, title, file_title}]
    """
    # 1. 定义切割正则 / md_content 按行切割
    #    ^\s*#{1,6}\s+.+ 匹配 [任意空白]###[任意空白]标题描述
    title_pattern = re.compile(r'^\s*#{1,6}\s+.+')
    lines = md_content.split('\n')
    # 准备存储数据容器
    chunks = []          # 最终结果
    current_title = ""   # 当前标题
    current_lines = []   # 当前标题下累计内容
    in_code_block = False  # 记录是否在代码块中
    title_count = 0

    # 2. 循环处理每行数据
    for line in lines:
        strip_line = line.strip()
        # 判断是否在代码块边界（``` 或 ~~~）
        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            in_code_block = not in_code_block  # 进入/退出取反
            current_lines.append(line)
            continue
        # 不在代码块内，才识别标题
        if not in_code_block and title_pattern.match(strip_line):
            # 遇到新标题，先把上一个标题的内容存起来
            if current_title:
                chunks.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            # 重置变量，开始记录当前标题块
            current_title = strip_line
            current_lines = [strip_line]
            title_count += 1
        else:
            # 普通内容行（标题行本身也保留在 content 里）
            current_lines.append(strip_line)

    # 3. 最后一块存储（循环结束后上一段还没保存）
    if current_title:
        chunks.append({
            "title": current_title.strip(),
            "content": "\n".join(current_lines),
            "file_title": file_title
        })

    # 4. 无标题兜底：整篇没有标题时，用「无主题」包成一块，内容不丢失
    if not chunks:
        chunks = [{
            "title": "无主题",
            "content": md_content,
            "file_title": file_title
        }]

    return chunks, title_count, len(lines)


@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections) -> List[Dict]:
    """
    同一标题下、同一语义单元内的二次超长切割。
    对超过 CHUNK_SIZE 的块用 RecursiveCharacterTextSplitter 递归切，
    不硬断句子、不产生碎片，并为每个子块补齐 parent_title / part 溯源信息。
    :param sections: 按标题初切的结果
    :return: 二次切割后的标准 chunk 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 切割优先级：段落 -> 换行 -> 中文句末标点 -> 空格
        separators=_SEPARATORS,
    )
    # 进行切割
    final_chunks = []
    for section in sections:
        # 二次切割标题下的内容
        sub_chunks = splitter.split_text(section["content"])
        has_multiple_chunks = len(sub_chunks) > 1
        # 生成带编号的子块
        for idx, chunk in enumerate(sub_chunks, start=1):
            current_title = f"{section['title']}_{idx}" if has_multiple_chunks else section["title"]
            final_chunks.append({
                "title": current_title,
                "content": chunk.strip(),
                "file_title": section["file_title"],
                "parent_title": section["title"],
                "part": idx
            })
    return final_chunks


@step_log("step_4_backup_chunks")
def step_4_backup_chunks(final_chunks, state):
    """
    将最终切分结果备份到本地（md 同级的 backup_chunks.json），
    同时 state 已在主流程里更新，这里只负责落盘持久化，方便调试与溯源。
    :param final_chunks: 要备份的 chunk 列表
    :param state: 提供 md_path 以定位备份目录
    :return: None
    """
    md_path = state.get('md_path') or ''
    # 兜底：没有 md_path 时放到项目根目录的 output 下
    backup_dir = Path(md_path).parent if md_path else (PROJECT_ROOT / "output")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file_path = backup_dir / "backup_chunks.json"

    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,  # 中文直接原文存储
            indent=4             # json 带 4 空格缩进，便于阅读
        )
    logger.debug(f"数据备份成功，备份地址：{backup_file_path}")


# =============== 单元测试 ===============
if __name__ == '__main__':
    """
    单元测试：联合 node_md_img（图片处理节点）进行集成测试
    测试条件：1. 已配置 .env  2. 存在测试 MD 文件
    测试流程：先运行图片处理 -> 再运行文档切分，验证端到端流程
    """
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试 MD 文件路径（笔记指定用 hak180，无图会被 node_md_img 跳过但会回填 md_content）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试 MD 文件放入项目根目录的 output 目录下")
        raise SystemExit(1)

    # 构造测试状态对象，模拟流程入参
    test_state = {
        "md_path": test_md_path,
        "task_id": "test_task_123456",
        "md_content": "",
        "file_title": "hak180产品安全手册",
        "local_dir": os.path.join(PROJECT_ROOT, "output"),
    }
    logger.info("开始本地测试 - MD 图片处理全流程")
    # 先执行图片处理节点（无图则跳过，但会回填 md_content）
    result_state = node_md_img(test_state)
    logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
    logger.info("\n=== 开始执行文档切分节点集成测试 ===")

    logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
    final_state = node_document_split(result_state)
    final_chunks = final_state.get("chunks", [])
    logger.info(f"✅ 测试成功：最终生成 {len(final_chunks)} 个有效 Chunk")
    for i, c in enumerate(final_chunks[:5], 1):
        logger.info(f"  Chunk {i}: title={c['title']!r} | content={c['content'][:40]!r}")
