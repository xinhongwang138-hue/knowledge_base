# -*- coding: utf-8 -*-
"""存量切片重切分 + 重算向量

为什么需要这个脚本
------------------
入库代码的切分粒度（`CHUNK_SIZE`）改过，但库里已有的切片不会自动跟着变：
租户化迁移脚本 `kb_rebuild_with_tenant.py` 是「源集合原样搬运、向量不重算」，
所以库里躺着的还是更早版本切出来的大切片。实测一批切片长 702~2142 字，
而代码里写的是 400 —— 代码与数据长期不一致，页面上看不出任何异常，
只有评估时才暴露：问「关闭后内部零件多少度」，答案所在的 1702 字大切片
在向量空间里被整节内容稀释，排名落在召回窗口之外，表现为「文档里有，却答不出」。

这个脚本把已有内容按当前代码重切一遍：
    读回切片 → 按标题重组小节 → 摊平 HTML 表格 → 复用 node_document_split.step_3_refine_chunks
    → 重新生成 BGE-M3 稠密/稀疏向量 → 写入新切片 → 删除旧切片

为什么复用生产代码而不是再写一套切分：
    重切的意义就是让数据和「现在导入一份新文档会得到的结果」一致。两套实现一旦漂移，
    下次评估又会分不清是代码问题还是数据问题。

用法（脚本不在容器的挂载目录里，先拷进去再执行）
    docker cp scripts/reindex_chunks.py kb-import-api:/app/reindex_chunks.py
    docker exec kb-import-api python /app/reindex_chunks.py --tenant ws_1 --dry-run
    docker exec kb-import-api python /app/reindex_chunks.py --tenant ws_1

只重切某几份文档：
    docker exec kb-import-api python /app/reindex_chunks.py --tenant ws_1 --doc "hak180产品安全手册"

注意
    · 只重切「明显大于当前 CHUNK_SIZE」的文档；切片之间有 CHUNK_OVERLAP 的重叠，
      重切时按「上一片后缀 == 下一片前缀」把重叠去掉再拼，所以重跑不会把切口处的
      文字拼两遍。确实需要重跑（例如改了切分规则）时加 --force。
    · 先写新切片再删旧切片：中途失败最多留下重复数据（可重跑），不会出现"删了没写"的空洞。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.milvus_config import milvus_config  # noqa: E402
from app.clients.milvus_utils import get_milvus_client  # noqa: E402
from app.core.logger import logger  # noqa: E402
from app.import_process.agent.nodes.node_document_split import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TABLE_MERGE_MAX,
    step_3_refine_chunks,
)
from app.import_process.agent.nodes.node_import_milvus import _derive_doc_id  # noqa: E402
from app.lm.embedding_utils import generate_embeddings  # noqa: E402
from app.utils.html_table_utils import html_tables_to_markdown  # noqa: E402
from app.utils.text_overlap_utils import strip_overlap  # noqa: E402

# 判定「已经不是当前切分粒度」的阈值：正常切片不会超过 CHUNK_SIZE，
# 留 1.5 倍余量避免因为个别不可切片段反复触发重切；
# 表格会被整体合并（见 node_document_split.TABLE_MERGE_MAX），这类切片本来就会超出
# CHUNK_SIZE，阈值必须跟上，否则每次重切都会把它判成「过时」再切一遍。
RESPLIT_THRESHOLD = max(int(CHUNK_SIZE * 1.5), TABLE_MERGE_MAX)


def fetch_doc_chunks(client, tenant_id: str, file_title: str) -> list:
    """
    读取某份文档的全部切片，按写入顺序（chunk_id 升序）返回

    :param client: MilvusClient
    :param tenant_id: 租户标识
    :param file_title: 文件名
    :return: 切片列表
    """
    from app.utils.escape_milvus_string_utils import escape_milvus_string

    rows = client.query(
        collection_name=milvus_config.chunks_collection,
        filter=(
            f'tenant_id == "{escape_milvus_string(tenant_id)}" '
            f'and file_title == "{escape_milvus_string(file_title)}"'
        ),
        output_fields=["chunk_id", "title", "parent_title", "part", "item_name", "doc_id", "content"],
        limit=16384,
    )
    return sorted(rows, key=lambda r: r["chunk_id"])


def _strip_overlap(acc: str, nxt: str) -> str:
    """
    去掉下一片开头与已累积文本尾部重叠的部分

    实现已抽到 app/utils/text_overlap_utils.py（生产切分拼表格切口时也要用），
    这里只固定本场景的重叠上限。为什么必须去重叠：切片之间带 CHUNK_OVERLAP，
    直接拼接会把切口处的文字拼两遍——不只是难看，重复片段会让向量偏向切口附近，
    也会让答案里出现重复句子。

    :param acc: 已累积的正文
    :param nxt: 下一片正文
    :return: 去掉重叠后的下一片
    """
    return strip_overlap(acc, nxt, max_overlap=CHUNK_OVERLAP * 3)


def rebuild_sections(rows: list) -> list:
    """
    把切片按标题重组成小节

    只在「相邻且标题相同」时才归为一节：同一份文档里可能出现两个同名小节
    （例如两处 `## 设备`），按标题全局归并会把相隔很远的内容拼到一起。

    :param rows: 已按 chunk_id 排序的切片
    :return: [{title, content, file_title}, ...]
    """
    sections = []
    current_key = None
    current_text = ""

    def _flush():
        if current_text:
            sections.append({"title": current_key or "", "content": current_text})

    for row in rows:
        # 优先级：父标题（小节名）> 当前标题（子块名，带 _序号 后缀）
        key = (row.get("parent_title") or row.get("title") or "").strip()
        content = row.get("content") or ""
        if current_key is None:
            current_key = key
        elif key != current_key:
            _flush()
            current_text = ""
            current_key = key
        current_text = content if not current_text else f"{current_text}\n{_strip_overlap(current_text, content)}"
    _flush()
    return sections


def reindex_document(client, tenant_id: str, file_title: str, dry_run: bool, force: bool = False) -> dict:
    """
    重切一份文档并替换其切片

    :param client: MilvusClient
    :param tenant_id: 租户标识
    :param file_title: 文件名
    :param dry_run: 只统计不写入
    :param force: 忽略「已符合当前粒度」的保护，强制重切
    :return: 统计信息
    """
    rows = fetch_doc_chunks(client, tenant_id, file_title)
    if not rows:
        return {"file_title": file_title, "skipped": "没有切片"}

    old_sizes = [len(r.get("content") or "") for r in rows]
    max_old = max(old_sizes)
    if max_old <= RESPLIT_THRESHOLD and not force:
        return {
            "file_title": file_title,
            "skipped": f"已是当前粒度（最长 {max_old} 字 ≤ {RESPLIT_THRESHOLD}），无需重切",
        }

    # 主体名：同一份文档理应只有一个，历史上出现过漂移，取众数并提示
    name_counter = Counter((r.get("item_name") or "").strip() for r in rows)
    item_name = name_counter.most_common(1)[0][0]
    if len(name_counter) > 1:
        logger.warning(
            f"[{file_title}] 同一文档内存在多个 item_name：{dict(name_counter)}，"
            f"统一采用出现最多的 '{item_name}'"
        )

    doc_id = next((r.get("doc_id") for r in rows if r.get("doc_id")), "") or _derive_doc_id(
        tenant_id, file_title
    )

    # 1. 重组小节并摊平 HTML 表格（表格不摊平就切不动，CHUNK_SIZE 形同虚设）
    sections = rebuild_sections(rows)
    for section in sections:
        section["content"] = html_tables_to_markdown(section["content"])
        section["file_title"] = file_title

    # 2. 复用生产切分逻辑
    new_chunks = step_3_refine_chunks(sections)
    for chunk in new_chunks:
        chunk["item_name"] = item_name
        chunk["doc_id"] = doc_id
        chunk["tenant_id"] = tenant_id
    new_sizes = [len(c["content"]) for c in new_chunks]

    stat = {
        "file_title": file_title,
        "old_count": len(rows),
        "old_max": max_old,
        "new_count": len(new_chunks),
        "new_max": max(new_sizes) if new_sizes else 0,
        "item_name": item_name,
    }
    logger.info(
        f"[{file_title}] 重切：{len(rows)} 片（最长 {max_old}）-> "
        f"{len(new_chunks)} 片（最长 {stat['new_max']}）"
    )
    if dry_run:
        stat["skipped"] = "dry-run，未写入"
        return stat

    # 3. 重算向量：输入文本格式必须与 node_bge_embedding 一致，
    #    否则新旧切片的向量不在同一语义空间，检索分数不可比
    texts = [
        f"商品：{c['item_name']}，介绍：{c['content']}" if c["item_name"] else c["content"]
        for c in new_chunks
    ]
    embeddings = generate_embeddings(texts)
    for index, chunk in enumerate(new_chunks):
        chunk["dense_vector"] = embeddings["dense"][index]
        chunk["sparse_vector"] = embeddings["sparse"][index]

    # 4. 先写新的，再删旧的
    insert_data = [
        {
            "content": c["content"],
            "title": c["title"],
            "parent_title": c.get("parent_title", ""),
            "part": int(c.get("part", 1)),
            "file_title": file_title,
            "item_name": c["item_name"],
            "tenant_id": tenant_id,
            "doc_id": c["doc_id"],
            "dense_vector": c["dense_vector"],
            "sparse_vector": c["sparse_vector"],
        }
        for c in new_chunks
    ]
    client.insert(collection_name=milvus_config.chunks_collection, data=insert_data)
    old_ids = [r["chunk_id"] for r in rows]
    client.delete(collection_name=milvus_config.chunks_collection, ids=old_ids)
    client.flush(collection_name=milvus_config.chunks_collection)
    stat["deleted_old"] = len(old_ids)
    logger.info(f"[{file_title}] 已写入 {len(insert_data)} 片，删除旧切片 {len(old_ids)} 条")
    return stat


def dedupe_document(client, tenant_id: str, file_title: str, dry_run: bool) -> dict:
    """
    删除同一文档内正文完全重复的切片（每组保留 chunk_id 最小的那条）

    什么时候会出现重复：重切过程中进程被中断（Ctrl-C、容器重启、超时被杀），
    新切片已经 insert 但还没 flush，也还没删旧切片。下一次跑 `fetch_doc_chunks`
    看不到那批未 flush 的数据，于是又写了一遍，flush 之后两批一起可见。
    这不是理论风险——本项目实测踩到过：21 条切片变成 42 条。

    :param client: MilvusClient
    :param tenant_id: 租户标识
    :param file_title: 文件名
    :param dry_run: 只统计不删除
    :return: 统计信息
    """
    rows = fetch_doc_chunks(client, tenant_id, file_title)
    if not rows:
        return {"file_title": file_title, "skipped": "没有切片"}

    seen = {}
    duplicates = []
    for row in rows:
        # rows 已按 chunk_id 升序，第一次遇到的就是要保留的那条
        key = (row.get("content") or "").strip()
        if key in seen:
            duplicates.append(row["chunk_id"])
        else:
            seen[key] = row["chunk_id"]

    if not duplicates:
        return {"file_title": file_title, "skipped": f"{len(rows)} 片无重复"}
    if not dry_run:
        client.delete(collection_name=milvus_config.chunks_collection, ids=duplicates)
        client.flush(collection_name=milvus_config.chunks_collection)
    logger.info(f"[{file_title}] 去重：删除 {len(duplicates)} 条重复切片"
                f"{'（dry-run 未执行）' if dry_run else ''}")
    return {
        "file_title": file_title,
        "dedupe_removed": 0 if dry_run else len(duplicates),
        "dedupe_planned": len(duplicates),
    }


def list_file_titles(client, tenant_id: str) -> list:
    """
    列出某租户下所有文档名（用于不带 --doc 时全量处理）

    :param client: MilvusClient
    :param tenant_id: 租户标识
    :return: 去重后的文件名列表
    """
    from app.utils.escape_milvus_string_utils import escape_milvus_string

    rows = client.query(
        collection_name=milvus_config.chunks_collection,
        filter=f'tenant_id == "{escape_milvus_string(tenant_id)}"',
        output_fields=["file_title"],
        limit=16384,
    )
    return list(dict.fromkeys(r.get("file_title") or "" for r in rows if r.get("file_title")))


def main():
    parser = argparse.ArgumentParser(description="存量切片按当前 CHUNK_SIZE 重切并重算向量")
    parser.add_argument("--tenant", required=True, help="租户标识，如 ws_1")
    parser.add_argument("--doc", action="append", default=None,
                        help="只处理指定文件（可重复），缺省处理该租户全部文档")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    parser.add_argument("--dedupe", action="store_true",
                        help="只做去重（清理中断重切留下的重复切片），不做重切")
    parser.add_argument("--force", action="store_true",
                        help="即使已符合当前粒度也重切（重叠区会在拼接时去掉，可安全重跑）")
    args = parser.parse_args()

    client = get_milvus_client()
    if client is None:
        raise SystemExit("Milvus 不可用，检查 MILVUS_URL 配置")
    client.load_collection(collection_name=milvus_config.chunks_collection)

    targets = args.doc or list_file_titles(client, args.tenant)
    if not targets:
        raise SystemExit(f"租户 {args.tenant} 下没有任何切片")

    logger.info(
        f"开始重切：tenant={args.tenant}, 目标 {len(targets)} 份文档, "
        f"CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}, dry_run={args.dry_run}"
    )

    stats = []
    for file_title in targets:
        try:
            if args.dedupe:
                # 去重后立刻出报告，不掺重切统计，避免两件事混在一起看不清
                stat = dedupe_document(client, args.tenant, file_title, args.dry_run)
            else:
                stat = reindex_document(client, args.tenant, file_title, args.dry_run, args.force)
        except Exception as exc:
            logger.error(f"[{file_title}] 处理失败：{exc}", exc_info=True)
            stat = {"file_title": file_title, "skipped": f"失败：{exc}"}
        stats.append(stat)

    print("\n===== 汇总 =====")
    for stat in stats:
        if stat.get("dedupe_planned") is not None:
            print(
                f"  {stat['file_title']}: 重复切片 {stat['dedupe_planned']} 条"
                f"{'（dry-run，未删除）' if args.dry_run else '，已删除'}"
            )
        elif stat.get("skipped"):
            print(f"  {stat['file_title']}: {stat['skipped']}")
        else:
            print(
                f"  {stat['file_title']}: {stat['old_count']} 片(最长 {stat['old_max']}) -> "
                f"{stat['new_count']} 片(最长 {stat['new_max']})"
            )


if __name__ == "__main__":
    main()
