# 【必须放在最前】限制 MKL/OpenMP 线程数，早于任何 torch 初始化，
# 否则 BGE-M3 向量化会 Segmentation fault。原理见 app/core/torch_env.py。
import app.core.torch_env  # noqa: F401

# 加载环境变量：从 .env 文件读取配置
from dotenv import load_dotenv

# 导入 LangGraph 核心依赖
from langgraph.graph import StateGraph, END

# 导入自定义状态类：统一管理工作流全程的所有数据
from app.import_process.agent.state import ImportGraphState

# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
# 入口节点：初始化参数，校验输入
from app.import_process.agent.nodes.node_entry import node_entry
# PDF转MD：解析 PDF 文件为 markdown 格式
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
# MD图片处理：提取/下载 markdown 中的图片，修复图片路径
from app.import_process.agent.nodes.node_md_img import node_md_img
# 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_document_split import node_document_split
# 主体名识别：从分块中提取核心主体名称(业务定制化)
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
# BGE向量化：将文本分块转换为向量表示(适配 Milvus 向量库)
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
# 导入 Milvus：将向量数据写入 Milvus 向量数据库
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus


# 初始化环境变量：必须在配置读取前执行，确保后续节点能获取到 .env 中的配置
# 注意：load_dotenv 默认 override=False —— 已存在的进程环境变量优先，容器注入值生效
load_dotenv()


# ================ 1. 初始化 LangGraph 状态图 ===============
# 核心：StateGraph 是 LangGraph 的核心类，用于构建有状态的工作流
# 参数 ImportGraphState：自定义 TypedDict 类型，定义了工作流的**全量状态字段**
# 作用：所有节点的入参都是该状态对象，节点返回的键值对会自动合并回状态，实现节点间数据共享
workflow = StateGraph(ImportGraphState)


# ================ 2. 注册所有业务节点 ============
# 语法：add_node("节点唯一标识", 节点函数)
# 要求：节点函数必须接收[状态对象]作为入参，返回字典（用于更新状态）
# 所有节点按[知识库导入流程]先后顺序注册，节点标识与函数名保持一致，便于维护
# 流程入口：参数初始化，输入校验
workflow.add_node("node_entry", node_entry)
# PDF转MD：非 MD 格式文件的前置处理
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
# MD图片处理：保证文档中图片的可访问性
workflow.add_node("node_md_img", node_md_img)
# 文档分块：解决大文本无法向量化/推理的问题
workflow.add_node("node_document_split", node_document_split)
# 主体名识别：业务定制化步骤，提取核心业务标识
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
# BGE向量化：文本 → 向量，为 Milvus 存储做准备
workflow.add_node("node_bge_embedding", node_bge_embedding)
# 向量入库：将向量数据持久化到 Milvus
workflow.add_node("node_import_milvus", node_import_milvus)


# =============== 3. 设置工作流入口节点 ===============
# 语法：set_entry_point("节点标识")→ 推荐写法，直接指定流程起始节点
# 等效写法：workflow.add_edge(START, "node_entry")（START 是 LangGraph 内置起始常量）
# 作用：指定工作流执行的第一个节点，替代手动添加 START 到目标节点的边，代码更简洁
workflow.set_entry_point("node_entry")


# ===== 4. 定义条件路由函数（入口节点后的分支逻辑）=======
# 核心：根据状态中的配置项，动态决定后续执行路径，实现 [PDF导入] / [MD直接导入] 分支
# 要求：接收状态对象为入参，返回[目标节点标识] 或 END（内置结束常量）
def route_after_entry(state: ImportGraphState) -> str:
    """
        入口节点后的条件路由逻辑
        :param state: 工作流全量状态对象，包含所有配置项和中间结果
        :return: 目标节点标识 / END，LangGraph 会自动跳转到对应节点
    """
    # 分支1：开启 MD 直接导入 → 跳过 PDF转MD，直接执行 MD图片处理
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # 分支2：开启 PDF 导入 → 执行 PDF转MD，再走后续流程
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # 分支3：未开启任何导入配置 → 直接终止工作流（END 是 LangGraph 内置结束常量）
    return END


# 注册条件边：将入口节点与路由函数绑定
# 语法：add_conditional_edges("源节点标识", 路由函数, 路径映射表)
# 作用：源节点执行完成后，调用路由函数，根据返回值动态跳转到目标节点
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)


# =============== 5. 定义固定流转边 ===============
# PDF转MD 之后统一进入 MD图片处理
workflow.add_edge("node_pdf_to_md", "node_md_img")
# 以下为导入主链路，逐级串联直到入库
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
# 入库完成后结束
workflow.add_edge("node_import_milvus", END)


# =============== 6. 编译工作流 ===============
# compile() 会校验节点名与边是否一致，此处若节点名写错会抛 ValueError
kb_import_app = workflow.compile()


# =============== 7. 主图全流程测试 ===============
if __name__ == "__main__":
    # 说明：笔记里写的 from app.utils.path_util import PROJECT_ROOT 在本项目不存在，
    #      PROJECT_ROOT 实际定义在 app.core.logger，故改从该处导入。
    #      另外笔记用 ImportGraphState({...}) 调用 TypedDict，这是不合法的（TypedDict 只是类型标注，
    #      不是可调用类），这里改为普通 dict，与项目内其它节点的测试写法保持一致。
    import os

    from app.core.logger import PROJECT_ROOT, logger
    from app.import_process.agent.state import create_default_state

    # 全流程测试：验证 PDF导入 -> 图片处理 -> 切分 -> 主体识别 -> 向量化 -> Milvus入库 完整链路
    logger.info("===== 开始执行知识库导入全流程测试 =====")

    # 1. 构造测试文件路径（复用项目 doc 目录）
    # 默认用「测试带图文档.pdf」：该文件含真实图片，可覆盖 md_img 节点的完整分支；
    # 如需换成纯文本 PDF，把下面这行改成 "hak180产品安全手册.pdf" 即可。
    test_pdf_name = os.path.join("doc", "测试带图文档.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    # 2. 构造输出目录（存放 MD / 图片等中间文件）
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # 不存在则创建

    # 3. 校验测试 PDF 文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试 PDF 文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的 doc 文件夹中")
    else:
        # 4. 构造测试状态
        # 注意：is_pdf_read_enabled / is_md_read_enabled 无需在此硬编码，
        #      node_entry 会依据文件后缀自动判定走哪条分支，强行写死反而会与实际路径冲突
        #      （笔记里写的是 False，注释却写「开启PDF解析」，自相矛盾）。
        test_state = create_default_state(
            task_id="test_kg_import_workflow_001",  # 测试任务ID
            local_file_path=test_pdf_path,          # 测试 PDF 文件路径
            local_dir=test_output_dir,               # 中间文件输出目录
        )
        try:
            logger.info(f"测试任务启动，PDF 文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus")

            # 5. 执行 LangGraph 全流程（流式执行，打印节点执行进度）
            final_state = None
            for step in kb_import_app.stream(test_state, stream_mode="values"):
                # 打印当前执行完成的节点（流式输出更直观）
                current_node = list(step.keys())[-1] if step else "未知节点"
                logger.info(f"✅ 节点执行完成：{current_node}")
                final_state = step  # 保存最终状态

            # 6. 全流程执行完成，结果预览和核心指标打印
            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = (final_state.get("md_content", "") or "")[:150]
                has_embedding = (
                    all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                )
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False

                # 打印核心指标
                logger.info(f"📄 PDF转MD 内容预览（前150 字符）：{md_content}...")
                logger.info(f"📝 文档切分总切片数：{chunk_count}")
                logger.info(f"🏷️  识别出的主体名称：{final_state.get('item_name', '未识别')}")
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"🗄️  所有切片是否完成 Milvus 入库（含 chunk_id）：{'是' if has_chunk_id else '否'}")
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")
                if chunks:
                    logger.info("-" * 80)
                    logger.info("===== 切片详情（前 3 条）=====")
                    for idx, c in enumerate(chunks[:3], 1):
                        dense_dim = len(c.get("dense_vector", []) or [])
                        sparse_n = len(c.get("sparse_vector", {}) or {})
                        logger.info(
                            f"  [{idx}] chunk_id={c.get('chunk_id')} title={c.get('title')!r} "
                            f"part={c.get('part')} dense维度={dense_dim} 稀疏特征数={sparse_n}"
                        )
                        logger.info(f"       内容预览：{(c.get('content') or '')[:60]}")
                logger.info("-" * 80)
        except Exception:
            logger.exception("===== 全流程测试运行失败 =====")
    logger.info("===== 知识库导入全流程测试结束 =====")
