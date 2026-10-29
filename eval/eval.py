"""
知识库检索效果离线评估（HTTP 版）

设计取舍（为什么不是照搬 Ragas 那套）：
    1. Ragas 的五项指标依赖 LLM 当裁判，同一份数据两次跑分数会有波动，
       而且要先装 ragas / datasets 一大包依赖，与本项目「本地可跑、结果可复现」的取向冲突；
    2. 本脚本改用**关键词命中**做确定性度量：测试集里每条样本除标准答案外，
       还给出 expect_keywords（必须出现的关键事实），分别统计
        · 上下文召回（context_recall）：命中事实是否出现在召回片段的 snippet 里
        · 答案覆盖（answer_recall）：命中事实是否出现在最终答案里
       两者一对比就能定位问题在「检索没找到」还是「找到了但没答出来」；
    3. 走 HTTP 调运行中的查询服务，而不是在进程内跑图 —— 本机是 4GB 显存，
       容器里已经常驻 BGE-M3 与 reranker，再在宿主加载一份必然 OOM；
       同时这也顺带验证了真实部署链路（含租户令牌、来源溯源）。

用法：
    .venv\\Scripts\\python.exe eval/eval.py                       # 全量用例
    .venv\\Scripts\\python.exe eval/eval.py --limit 3             # 先跑 3 条冒烟
    .venv\\Scripts\\python.exe eval/eval.py --tenant ws_1         # 指定租户（默认 ws_1）
    .venv\\Scripts\\python.exe eval/eval.py --out eval/result.csv

前置条件：
    · 查询服务已启动（默认 http://127.0.0.1:8011，docker compose 里的 kb-query-api）
    · 若服务端 WEB_FALLBACK_ENABLED=1，本地分数不足时会触发联网兜底，
      评估结果会混入网络内容；要纯净的本地检索评估，先把服务端该变量设为 0 再重启。
"""
import argparse
import csv
import hashlib
import hmac
import json
import os
import sys
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

# 用绝对路径加载 .env，保证脚本在任意工作目录下都能拿到密钥配置
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_QA = EVAL_DIR / "qa.csv"
DEFAULT_OUT = EVAL_DIR / "result_latest.csv"
DEFAULT_BASE = "http://127.0.0.1:8011"
DEFAULT_TENANT = "ws_1"


# ============================ 请求头（令牌 / Key） ============================
def _b64(raw: bytes) -> str:
    """base64url 编码并去掉填充，与 app/utils/tenant_guard.py 的签发格式保持一致"""
    return urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def build_internal_token(tenant: str, secret: str, ttl: int = 1800) -> str:
    """
    按智库侧的签发格式现造一个内部令牌

    格式：base64url(payload).base64url(HMAC-SHA256(payload, secret))，
    payload 至少含 tenant 与 exp —— 与电商后端签给智库的令牌同构。

    :param tenant: 租户标识
    :param secret: KB_INTERNAL_SECRET
    :param ttl: 有效期（秒）
    :return: 可直接放进 X-KB-Internal 头的令牌
    """
    payload = _b64(
        json.dumps({"tenant": tenant, "exp": int(time.time()) + ttl}).encode("utf-8")
    )
    signature = _b64(
        hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    )
    return f"{payload}.{signature}"


def build_headers(tenant: str) -> Dict[str, str]:
    """
    构造调用查询服务所需的请求头

    · X-API-Key：服务端配置了 WRITE_API_KEY 时必带（写接口校验）
    · X-KB-Internal：服务端 KB_REQUIRE_TENANT=true 时必带，租户取自入参

    :param tenant: 租户标识
    :return: 请求头字典
    """
    headers = {"Content-Type": "application/json"}
    api_key = (os.getenv("WRITE_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    require_tenant = (os.getenv("KB_REQUIRE_TENANT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    secret = (os.getenv("KB_INTERNAL_SECRET") or "").strip()
    if require_tenant and secret:
        headers["X-KB-Internal"] = build_internal_token(tenant, secret)
    return headers


# ============================ 测试集与指标 ============================
def load_cases(path: Path) -> List[Dict[str, str]]:
    """
    读取测试集（question / ground_truth / expect_keywords）

    :param path: qa.csv 路径
    :return: 用例列表
    """
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cases = []
    for row in rows:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        keywords = [
            k.strip()
            for k in (row.get("expect_keywords") or "").replace(";", "|").split("|")
            if k.strip()
        ]
        cases.append(
            {
                "question": question,
                "ground_truth": (row.get("ground_truth") or "").strip(),
                "keywords": keywords,
            }
        )
    return cases


def keyword_recall(text: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    关键词召回率：命中数 / 总数

    :param text: 待检查文本（召回片段拼接 / 答案正文）
    :param keywords: 期望出现的关键事实
    :return: (召回率, 命中词, 缺失词)；无关键词时返回 1.0
    """
    if not keywords:
        return 1.0, [], []
    haystack = (text or "").lower()
    hit = [k for k in keywords if k.lower() in haystack]
    miss = [k for k in keywords if k.lower() not in haystack]
    return len(hit) / len(keywords), hit, miss


def run_case(
    case: Dict[str, Any], index: int, base_url: str, headers: Dict[str, str], timeout: int
) -> Dict[str, Any]:
    """
    跑一条用例：调 /query（非流式）→ 计算指标 → 清理该会话历史

    :param case: 用例
    :param index: 序号（用于生成隔离的 session_id）
    :param base_url: 查询服务地址
    :param headers: 请求头
    :param timeout: 单条超时（秒）
    :return: 指标字典
    """
    # 每条用例独立会话：避免上一轮的问答被当成上下文喂给下一轮（多轮历史会污染指标）
    session_id = f"eval-{int(time.time() * 1000)}-{index}"
    started = time.time()
    result: Dict[str, Any] = {
        "session_id": session_id,
        "question": case["question"],
        "answer": "",
        "doc_recall": 0.0,
        "answer_recall": 0.0,
        "top1_score": "",
        "sources": 0,
        "web": False,
        "clarify": False,
        "source_titles": "",
        "item_names": "",
        "message_id": "",
        "latency_ms": 0,
        "hit_keywords": "",
        "miss_keywords": "",
        "error": "",
    }

    try:
        resp = requests.post(
            f"{base_url}/query",
            headers=headers,
            json={
                "query": case["question"],
                "session_id": session_id,
                "is_stream": False,
                # 让服务把真正进入 Prompt 的精排上下文一起返回，
                # 否则只能拿 200 字来源摘要算召回，长切片后段的关键词会漏算
                "debug_context": True,
            },
            timeout=timeout,
        )
        payload = resp.json() if resp.content else {}
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {str(payload)[:200]}")
    except Exception as exc:  # 单条失败不该中断整轮评估
        result["latency_ms"] = int((time.time() - started) * 1000)
        result["error"] = str(exc)[:200]
        return result

    sources = payload.get("sources") or []
    contexts = payload.get("contexts") or []
    answer = payload.get("answer") or ""
    # 召回侧的检查文本：优先用精排后的完整上下文（真正进入 Prompt 的内容），
    # 拿不到时退回来源摘要（snippet 只截 200 字，会低估召回）
    if contexts:
        context_text = "\n".join(
            " ".join(
                str(c.get(field) or "")
                for field in ("file_title", "title", "content")
            )
            for c in contexts
        )
    else:
        context_text = "\n".join(
            " ".join(
                str(s.get(field) or "")
                for field in ("title", "parent_title", "file_title", "snippet")
            )
            for s in sources
        )

    doc_recall, hit, miss = keyword_recall(context_text, case["keywords"])
    ans_recall, _, _ = keyword_recall(answer, case["keywords"])
    scores = [s.get("score") for s in sources if isinstance(s.get("score"), (int, float))]

    result.update(
        {
            "answer": answer,
            "doc_recall": round(doc_recall, 4),
            "answer_recall": round(ans_recall, 4),
            "top1_score": round(max(scores), 4) if scores else "",
            # 优先报精排上下文条数（真正进入 Prompt 的），没有则用来源摘要条数
            "sources": len(contexts) if contexts else len(sources),
            "web": any((s.get("source") or "local") == "web" for s in sources),
            # 反问识别：主体歧义时节点会直接回「您是想问以下哪个产品」，
            # 这类样本没走检索，召回分数必须单列统计，否则会把均值拉低成假信号
            "clarify": "您是想问以下哪个产品" in answer,
            # 来源标题：0 召回时靠它判断「是没检索到」还是「检索到别的文档」
            "source_titles": " | ".join(
                f"{s.get('file_title') or s.get('title') or '?'}"
                f"（{(s.get('score') if s.get('score') is not None else '-')}）"
                for s in sources
            ),
            "item_names": "、".join(payload.get("item_names") or []),
            "message_id": payload.get("message_id") or "",
            "latency_ms": int((time.time() - started) * 1000),
            "hit_keywords": "、".join(hit),
            "miss_keywords": "、".join(miss),
        }
    )

    # 评估不该在用户的会话列表里留垃圾：跑完即清（失败也不影响指标）
    try:
        requests.delete(f"{base_url}/history/{session_id}", headers=headers, timeout=10)
    except Exception:
        pass
    return result


# ============================ 输出 ============================
def write_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    """
    结果落盘（utf-8-sig，Excel 直接打开不乱码）

    :param path: 输出路径
    :param results: 逐条结果
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "session_id", "question", "answer", "doc_recall", "answer_recall",
        "top1_score", "sources", "web", "clarify", "source_titles", "item_names",
        "message_id", "latency_ms", "hit_keywords", "miss_keywords", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    控制台摘要：均值 + 完全召回率 + 最差用例，便于快速判断这轮改动是涨还是跌

    :param results: 逐条结果
    """
    ok = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    print("\n" + "=" * 78)
    if not ok:
        print(f"全部 {len(results)} 条均失败，先确认服务可用：{failed[0]['error'] if failed else ''}")
        print("=" * 78)
        return

    # 反问样本没走检索（主体歧义时节点直接回「您想问哪个产品」），
    # 单独计数、不参与召回均值，否则会把「数据治理问题」算成「检索质量问题」
    clarify = [r for r in ok if r["clarify"]]
    scored = [r for r in ok if not r["clarify"]]
    if not scored:
        print(f"{len(ok)} 条全部被主体确认节点反问，无有效检索样本；请把问题写得更具体（带型号）")
        print("=" * 78)
        return

    n = len(scored)
    doc_mean = sum(r["doc_recall"] for r in scored) / n
    ans_mean = sum(r["answer_recall"] for r in scored) / n
    full = sum(1 for r in scored if r["doc_recall"] >= 0.999)
    both = sum(1 for r in scored if r["doc_recall"] >= 0.999 and r["answer_recall"] >= 0.999)
    latency = sum(r["latency_ms"] for r in ok) / len(ok) / 1000
    web = sum(1 for r in ok if r["web"])
    scores = [r["top1_score"] for r in scored if isinstance(r["top1_score"], (int, float))]

    print(f"用例 {len(ok)} 条（失败 {len(failed)}，其中反问 {len(clarify)} 条不计入召回）")
    print(f"上下文召回均值 {doc_mean:.3f}   答案覆盖均值 {ans_mean:.3f}")
    print(f"上下文完全召回 {full}/{n}   检索与答案双命中 {both}/{n}")
    score_text = f"   平均 Top1 分 {sum(scores)/len(scores):.3f}" if scores else ""
    print(f"平均耗时 {latency:.2f}s   触发联网 {web} 次{score_text}")
    if clarify:
        print(f"被反问的问题：{'；'.join(r['question'][:18] for r in clarify[:4])}")

    worst = sorted(scored, key=lambda r: (r["doc_recall"], r["answer_recall"]))[:3]
    print("\n最差 3 条：")
    for r in worst:
        print(f"  · {r['question']}")
        print(f"    上下文 {r['doc_recall']:.2f} / 答案 {r['answer_recall']:.2f}，缺失：{r['miss_keywords'] or '无'}")
        if r["source_titles"]:
            print(f"    实际来源：{r['source_titles'][:90]}")
    if failed:
        print("\n失败用例：")
        for r in failed[:3]:
            print(f"  · {r['question']} -> {r['error']}")
    print("=" * 78)


def main() -> None:
    """命令行入口：解析参数 → 逐条评估 → 落盘 + 摘要"""
    parser = argparse.ArgumentParser(description="知识库检索效果离线评估（HTTP 版）")
    parser.add_argument("--qa", default=str(DEFAULT_QA), help="测试集 CSV 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="结果 CSV 输出路径")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="查询服务地址")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="租户标识（默认 ws_1）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    parser.add_argument("--timeout", type=int, default=180, help="单条请求超时（秒）")
    args = parser.parse_args()

    # Windows 控制台默认 GBK，输出中文与分隔线会乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = load_cases(Path(args.qa))
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"测试集为空：{args.qa}")

    headers = build_headers(args.tenant)
    print(f"评估开始：{len(cases)} 条用例 → {args.base_url}（租户 {args.tenant}）")
    print(f"请求头：{sorted(k for k in headers if k != 'Content-Type')}")

    results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        row = run_case(case, i, args.base_url, headers, args.timeout)
        results.append(row)
        mark = "OK " if not row["error"] else "ERR"
        print(
            f"[{i:>2}/{len(cases)}] {mark} 召回 {row['doc_recall']:.2f} "
            f"答案 {row['answer_recall']:.2f} {row['latency_ms']:>6}ms  {case['question'][:34]}"
        )

    write_csv(Path(args.out), results)
    print_summary(results)
    print(f"明细已写入：{args.out}")


if __name__ == "__main__":
    main()
