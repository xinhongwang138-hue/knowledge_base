# -*- coding: utf-8 -*-
"""
存量集合租户化迁移脚本（只读源集合、写新集合，不删除任何源数据）

背景：
    租户隔离改造后，写入侧（node_import_milvus / node_item_name_recognition）会建
    tenant_id 字段并在插入时强制写入，但改造前创建的 kb_chunks / kb_item_names
    没有该字段。检索侧强制拼 `tenant_id == "local"` 会让 Milvus 报
    "field tenant_id not exist"，节点 except 降级为空结果 —— 表现为本地召回恒为 0。

策略：
    1. 新建 kb_chunks_v2 / kb_item_names_v2（schema 与写入侧完全一致，含 tenant_id）
    2. 逐条搬运源集合数据，回填 tenant_id=local、doc_id=sha1("local|file_title")
    3. 向量原样复制，不重算（dense 1024 维 / sparse 稀疏向量）
    4. 源集合完全不改动，配置切回旧集合即可回滚

用法：
    python scripts/kb_rebuild_with_tenant.py --dry-run   # 只统计
    python scripts/kb_rebuild_with_tenant.py            # 真正写入
"""
import argparse
import hashlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from pymilvus import DataType  # noqa: E402

from app.clients.milvus_config import milvus_config  # noqa: E402
from app.clients.milvus_utils import get_milvus_client  # noqa: E402

# 迁移对象固定为租户化改造前的存量集合名，不随 .env 中的当前集合配置漂移，
# 否则配置已切到 *_v2 时会误建 *_v2_v2。
SRC_CHUNKS = "kb_chunks"
DST_CHUNKS = SRC_CHUNKS + "_v2"
SRC_ITEM = "kb_item_names"
DST_ITEM = SRC_ITEM + "_v2"
TENANT = "local"
BATCH = 500

CHUNK_FIELDS = [
    "content", "title", "parent_title", "part",
    "file_title", "item_name", "dense_vector", "sparse_vector",
]
ITEM_FIELDS = ["file_title", "item_name", "dense_vector", "sparse_vector"]


def derive_doc_id(tenant_id: str, file_title: str) -> str:
    key = f"{(tenant_id or '').strip()}|{(file_title or '').strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _has_field(client, collection: str, field: str) -> bool:
    desc = client.describe_collection(collection_name=collection)
    fields = desc.get("fields") if isinstance(desc, dict) else getattr(desc, "fields", [])
    return any(
        (f.get("name") if isinstance(f, dict) else getattr(f, "name", None)) == field
        for f in fields
    )


def ensure_chunks_collection(client) -> None:
    if client.has_collection(collection_name=DST_CHUNKS):
        print(f"[skip] {DST_CHUNKS} 已存在")
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_partition_key=True,
        num_partitions=64,
    )
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 32, "efConstruction": 300},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=DST_CHUNKS, schema=schema, index_params=index_params
    )
    print(f"[create] {DST_CHUNKS}")


def ensure_item_collection(client) -> None:
    if client.has_collection(collection_name=DST_ITEM):
        print(f"[skip] {DST_ITEM} 已存在")
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_partition_key=True,
        num_partitions=64,
    )
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=DST_ITEM, schema=schema, index_params=index_params
    )
    print(f"[create] {DST_ITEM}")


def migrate(client, src: str, dst: str, fields, tag: str, dry_run: bool) -> int:
    if not client.has_collection(collection_name=src):
        print(f"[warn] 源集合 {src} 不存在，跳过")
        return 0
    if not dry_run:
        client.load_collection(collection_name=src)
    rows = client.query(
        collection_name=src, filter="", output_fields=fields, limit=16384
    )
    print(f"[read] {src} -> {len(rows)} 条")
    if dry_run or not rows:
        return len(rows)

    prepared = []
    for row in rows:
        item = {k: row.get(k) for k in fields if row.get(k) is not None}
        item["tenant_id"] = TENANT
        if tag == "chunks":
            item["doc_id"] = derive_doc_id(TENANT, item.get("file_title"))
            if item.get("part") is None:
                item["part"] = 0
        prepared.append(item)

    # 幂等：先清掉本租户的旧目标数据
    try:
        client.load_collection(collection_name=dst)
        client.delete(
            collection_name=dst, filter=f'tenant_id == "{TENANT}"'
        )
        print(f"[clean] 已清理 {dst} 中 tenant={TENANT} 的旧数据")
    except Exception as exc:  # noqa: BLE001 清理失败不阻断，插入前人工确认即可
        print(f"[warn] 清理 {dst} 失败（继续插入）：{exc}")

    inserted = 0
    for i in range(0, len(prepared), BATCH):
        chunk = prepared[i:i + BATCH]
        res = client.insert(collection_name=dst, data=chunk)
        inserted += res.get("insert_count", 0)
    client.flush(collection_name=dst)
    print(f"[insert] {dst} 写入 {inserted} 条")
    return inserted


def main() -> None:
    global SRC_CHUNKS, DST_CHUNKS, SRC_ITEM, DST_ITEM, TENANT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--src-chunks", default=SRC_CHUNKS, help="源 chunks 集合名")
    parser.add_argument("--src-items", default=SRC_ITEM, help="源 item_name 集合名")
    parser.add_argument("--tenant", default=TENANT, help="回填的 tenant_id")
    args = parser.parse_args()

    SRC_CHUNKS = args.src_chunks
    DST_CHUNKS = SRC_CHUNKS + "_v2"
    SRC_ITEM = args.src_items
    DST_ITEM = SRC_ITEM + "_v2"
    TENANT = args.tenant

    client = get_milvus_client()
    if client is None:
        raise SystemExit("Milvus 客户端不可用，请检查 MILVUS_URL")

    print(f"milvus = {milvus_config.milvus_url}")
    print(f"chunks: {SRC_CHUNKS} -> {DST_CHUNKS}")
    print(f"items : {SRC_ITEM} -> {DST_ITEM}")
    print(f"tenant= {TENANT}  dry_run={args.dry_run}\n")

    if not args.dry_run:
        ensure_chunks_collection(client)
        ensure_item_collection(client)

    n1 = migrate(client, SRC_CHUNKS, DST_CHUNKS, CHUNK_FIELDS, "chunks", args.dry_run)
    n2 = migrate(client, SRC_ITEM, DST_ITEM, ITEM_FIELDS, "items", args.dry_run)

    if not args.dry_run:
        print(f"\n[verify] {DST_CHUNKS} rows = {client.get_collection_stats(DST_CHUNKS).get('row_count')}")
        print(f"[verify] {DST_ITEM} rows = {client.get_collection_stats(DST_ITEM).get('row_count')}")
        sample = client.query(
            collection_name=DST_CHUNKS,
            filter=f'tenant_id == "{TENANT}"',
            output_fields=["file_title", "item_name", "tenant_id", "doc_id"],
            limit=3,
        )
        for s in sample:
            print(f"[sample] {s}")
    else:
        print(f"\n[dry-run] 预计迁移 chunks={n1}, items={n2}")


if __name__ == "__main__":
    main()
