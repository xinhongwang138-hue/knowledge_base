import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

import requests

from app.core.logger import PROJECT_ROOT, logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.import_process.config.mineru_config import mineru_config
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    # 1.注解日志和进行和完成任务状态处理
    add_running_task(state['task_id'], 'node_pdf_to_md')
    # 2. step_1_validate_paths 校验路径完整以及是否真实存在
    pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
    # 3. step_2_upload_and_poll
    zip_url = step_2_upload_and_poll(pdf_path_obj)
    # 4. step_3_download_and_extract
    md_path = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
    # 5. 处理响应结果 state进行赋值处理
    state['md_path'] = md_path
    state['local_dir'] = str(output_dir_obj)
    # Path方案 使用read_text()
    state['md_content'] = Path(md_path).read_text(encoding='utf-8')
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state


@step_log("step_1 校验路径")
def step_1_validate_paths(state: ImportGraphState) -> Tuple[Path, Path]:
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性，遵循「输入严格校验、输出自动修复」的鲁棒性设计原则：
    1. 校验PDF路径非空且文件真实存在，不存在则直接抛出异常（快速失败）
    2. 校验输出目录，为空则赋予默认值，不存在则自动创建（自动容错）
    3. 统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性
    :param state: 流程状态字典，包含pdf_path、local_dir
    :return: (pdf_path_obj, local_dir_obj) 两个可直接使用的 Path 对象
    """
    # 1. 获取路径参数（strip 掉可能存在的首尾空白，避免 " x.pdf " 查不到文件）
    pdf_path = (state.get("pdf_path") or state.get("local_file_path") or "").strip()
    local_dir = (state.get("local_dir") or "").strip()

    # 2. 参数非空校验
    if not pdf_path:
        raise ValueError("pdf_path 不能为空，请提供有效的PDF文件路径")

    if not local_dir:
        local_dir = str(PROJECT_ROOT / "output")
        state["local_dir"] = local_dir
        logger.warning(f"未指定输出目录，使用默认路径：{local_dir}")

    # 3. 统一转换为Path对象，标准化路径处理
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 4. 路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(
            f"PDF文件不存在：{pdf_path_obj}，请检查文件路径是否正确"
        )
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"路径指向的不是文件：{pdf_path_obj}")

    # MinerU 服务端限制 200MB，超限服务端会直接拒，先在本地拦一道
    size_mb = pdf_path_obj.stat().st_size / 1024 / 1024
    if size_mb > mineru_config.max_file_mb:
        raise ValueError(
            f"文件超过 MinerU 限制：{size_mb:.1f}MB > {mineru_config.max_file_mb}MB"
        )

    if not local_dir_obj.exists():
        logger.warning(f"输出目录不存在，自动创建：{local_dir_obj}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


@step_log("step_2 上传并轮询")
def step_2_upload_and_poll(pdf_path_obj: Path) -> str:
    """
    step_2: 与 MinerU 交互（申请链接 → 上传 → 轮询）
    参数：pdf_path_obj(Path)
    返回：zip_url(str)
    1. 申请上传地址（同时拿到 batch_id）
    2. 文件上传 requests.put
    3. 按 batch_id 轮询取回 zip_url（限制最大等待时间，5xx 视为可重试的临时故障）
    4. 返回地址即可
    """
    upload_url, batch_id = _apply_upload_url(pdf_path_obj.name, pdf_path_obj.stem)
    _upload_file(upload_url, pdf_path_obj)
    return _poll_result(batch_id, pdf_path_obj.name)


def _apply_upload_url(file_name: str, data_id: str) -> Tuple[str, str]:
    """向 MinerU 申请上传地址，返回 (上传地址, batch_id)"""
    resp = requests.post(
        mineru_config.batch_submit_url,
        headers=mineru_config.auth_headers,
        json={
            "files": [{"name": file_name, "data_id": data_id}],
            # vlm 模型对表格、公式、版面还原更好，适合知识库场景
            "model_version": "vlm",
        },
        timeout=30,
    )
    _raise_for_business(resp, "申请上传链接")
    data = resp.json()["data"]
    file_urls = data.get("file_urls") or []
    batch_id = data.get("batch_id")
    if not file_urls:
        raise RuntimeError("MinerU 未返回上传地址")
    if not batch_id:
        # 没有 batch_id 就无法轮询，属于异常响应
        raise RuntimeError(f"MinerU 未返回 batch_id，响应内容：{data}")
    return file_urls[0], batch_id


def _upload_file(upload_url: str, pdf_path_obj: Path) -> None:
    """
    把本地文件 PUT 到预签名地址。上传后 MinerU 会自动创建解析任务，无需再提交

    ⚠️ 这里必须用独立的 Session 且 trust_env=False，原因：
    预签名 URL 直连对象存储(OSS/S3)，有字节级的签名校验。
    若走系统代理，代理会在 Header 里加东西(如 Via、Connection: close)，
    服务端发现请求头与签名约定不符，会判定请求被篡改并报
    403 SignatureDoesNotMatch。trust_env=False 强制绕过代理，
    发出纯净的原始请求。
    （POST/GET 是业务接口，Token 对了就行，不需要这么讲究）
    """
    with requests.Session() as session:
        session.trust_env = False
        with pdf_path_obj.open("rb") as fh:
            resp = session.put(
                upload_url,
                data=fh,
                headers={"Content-Length": str(pdf_path_obj.stat().st_size)},
                timeout=300,
            )
    resp.raise_for_status()
    logger.info(f"已上传：{pdf_path_obj.name}")


def _poll_result(batch_id: str, file_name: str) -> str:
    """
    轮询直到解析完成，返回结果 zip 的下载地址

    容错策略：
        - 200正常返回
        - 4xx 视为客户端问题（token 失效等），立刻失败，不浪费重试
        - 5xx / 网络抖动 视为临时故障，最多容忍 3 次连续失败
    """
    deadline = time.time() + mineru_config.timeout
    elapsed = 0
    net_errors = 0
    max_net_errors = 3
    # 轮询地址必须带上 batch_id，否则服务端返回 404
    poll_url = mineru_config.result_url(batch_id)

    while time.time() < deadline:
        try:
            resp = requests.get(
                poll_url,
                headers=mineru_config.auth_headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            net_errors += 1
            if net_errors > max_net_errors:
                raise RuntimeError(
                    f"查询解析结果连续失败 {net_errors} 次（含首次共 {net_errors} 次请求）：{exc}"
                ) from exc
            logger.warning(
                f"查询解析结果网络异常（{net_errors}/{max_net_errors}），"
                f"{mineru_config.poll_interval}s 后重试：{exc}"
            )
            time.sleep(mineru_config.poll_interval)
            continue

        if resp.status_code >= 500:
            net_errors += 1
            if net_errors > max_net_errors:
                raise RuntimeError(
                    f"MinerU 服务持续异常（HTTP {resp.status_code}，"
                    f"已容忍 {max_net_errors} 次重试后放弃）"
                )
            logger.warning(
                f"MinerU 返回 HTTP {resp.status_code}（{net_errors}/{max_net_errors}），"
                f"{mineru_config.poll_interval}s 后重试"
            )
            time.sleep(mineru_config.poll_interval)
            continue

        # 走到这说明网络与服务端都正常，重置连续失败计数
        net_errors = 0
        resp.raise_for_status()  # 4xx 在这里抛出，属于客户端问题不重试

        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"查询解析结果失败：{body.get('msg')}")

        results = body["data"].get("extract_result") or []
        item = next((r for r in results if r.get("file_name") == file_name), None)
        if item is None:
            # 结果列表为空是正常的：文件刚上传，服务端可能还没创建任务。
            # 若列表非空却找不到同名文件，那是真的出问题了。
            if not results:
                elapsed += mineru_config.poll_interval
                logger.info(
                    f"  等待服务端创建任务… 已等待 {elapsed}s"
                )
                time.sleep(mineru_config.poll_interval)
                continue
            raise RuntimeError(
                f"查询结果中找不到文件：{file_name}（返回了 {[r.get('file_name') for r in results]}）"
            )

        state = item.get("state")
        # waiting-file 在本流程里属正常：上传后服务端需要时间感知
        if state in ("pending", "running", "converting", "waiting-file"):
            elapsed += mineru_config.poll_interval
            logger.info(f"  MinerU 解析中… 状态={state} 已等待 {elapsed}s")
            time.sleep(mineru_config.poll_interval)
            continue

        if state == "failed":
            raise RuntimeError(f"MinerU 解析失败：{item.get('err_msg')}")

        if state == "done":
            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise RuntimeError("解析完成但未返回结果下载地址")
            return zip_url

        raise RuntimeError(f"未知的解析状态：{state}")

    raise TimeoutError(
        f"MinerU 解析超时（>{mineru_config.timeout}s），任务可能仍在服务端运行"
    )


def _raise_for_business(resp: requests.Response, action: str) -> None:
    """先查 HTTP 层错误，再查业务层 code"""
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"{action}失败：HTTP {resp.status_code} 认证被拒绝。\n"
            f"  请检查 .env 中的 MINERU_API_TOKEN 是否正确、是否已过期"
            f"（token 有效期 90 天），申请地址 https://mineru.net/apiManage/token\n"
            f"  当前 token 前缀：{resp.request.headers.get('Authorization', '')[:14]}…"
        )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"{action}失败：{body.get('msg')}")


@step_log("step_3 下载并解压")
def step_3_download_and_extract(zip_url: str, out_dir_obj: Path, stem: str) -> str:
    """
    step_3: 下载结果包并解压出 markdown
    参数：zip_url(str), out_dir_obj(Path), stem(原文件名去后缀)
    返回：解压后的 .md 的绝对路径(str)
    1. 下载 zip 到 output/{stem}_result.zip
    2. 解压目标目录 output/{stem}，已存在则先清空，防重复堆叠脏文件
    3. MinerU 结果包里的文件名不固定(full.md / 与原文件同名 / 其它)，
       这里统一重命名为 {stem}.md，避免同名冲突
    4. 返回绝对路径字符串
    """
    # 净化 stem：它是拿来做目录名和文件名的，不能含分隔符或非法字符。
    # 若以斜杠开头，Path 的 / 运算会丢掉 base 变成别的绝对路径，写到非预期位置。
    safe_stem = _sanitize_name(stem)
    if safe_stem != stem:
        logger.warning(f"文件名已净化：{stem!r} -> {safe_stem!r}")

    # 1. 下载 zip
    zip_path = out_dir_obj / f"{safe_stem}_result.zip"
    resp = requests.get(zip_url, stream=True, timeout=300)
    resp.raise_for_status()
    with zip_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 8192):
            fh.write(chunk)
    logger.info(f"已下载结果包：{zip_path.name}")

    # 2. 防重复：目标目录存在就先清空，避免残留旧文件混入
    extract_dir = out_dir_obj / safe_stem
    if extract_dir.exists():
        logger.info(f"输出目录已存在，先清空：{extract_dir}")
        try:
            shutil.rmtree(extract_dir)
        except OSError as exc:
            # 清空失败不应中断整个流程：目录已存在时 extractall 会覆盖同名文件，
            # 真正致命的是后面的 md 定位失败，这里只降级为警告。
            # （Windows 回收站不可用时 rmtree 会抛 OSError，必须兜住，
            #   否则会把「删除失败」顶替成「PDF 转换失败」，掩盖真实原因）
            logger.warning(f"清空旧输出目录失败，将直接覆盖写入：{exc}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 3. 解压
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    try:
        zip_path.unlink(missing_ok=True)  # zip 只是中间产物，解压完就删
    except OSError as exc:
        logger.warning(f"中间产物 zip 删除失败（不影响结果）：{exc}")

    # 4. 定位 markdown 并重命名统一
    md_file = _locate_and_rename_md(extract_dir, safe_stem)

    images_dir = extract_dir / "images"
    if images_dir.is_dir():
        logger.info(f"结果含 {len(list(images_dir.iterdir()))} 张图片，交给下游 node_md_img")

    # 5. 返回绝对路径
    return str(md_file.resolve())


def _sanitize_name(name: str) -> str:
    """
    净化文件名/目录名：去掉路径分隔符与 Windows 非法字符

    stem 会被拿来做目录名和文件名，必须净化：
      - 以斜杠开头时，Path 的 / 运算会丢掉 base，变成别的绝对路径
      - Windows 文件名不能含 : * ? " < > |，含了会直接创建失败
    """
    # 先取最后一段，防止 a/b/c 这种带目录的输入
    cleaned = Path(name).name
    # 替换 Windows 非法字符
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", cleaned).strip().rstrip(".")
    return cleaned or "document"


def _locate_and_rename_md(extract_dir: Path, stem: str) -> Path:
    """
    在解压结果里找到 markdown 并统一重命名为 {stem}.md

    按优先级找：与原文件同名 → full.md → 任意 .md。
    （用 rglob 而非 glob，因为 MinerU 可能把 md 放在子目录里）
    """
    md_files = [p for p in extract_dir.rglob("*.md") if not p.name.startswith(".")]
    if not md_files:
        raise FileNotFoundError(f"未找到PDF对应的MD文件：{extract_dir}")

    # 1) 优先与原 PDF 同名的 md
    target = extract_dir / f"{stem}.md"
    if not target.exists():
        same_name = next((p for p in md_files if p.stem == stem), None)
        # 2) 其次 full.md（MinerU 的标准产物）
        full_md = next((p for p in md_files if p.name.lower() == "full.md"), None)
        source = same_name or full_md or md_files[0]
        # with_name 只改文件名不动磁盘，rename 才真正改
        source.rename(source.with_name(target.name))
        logger.info(f"已重命名：{source.name} -> {target.name}")

    return target


# =============== 单元测试 ===============
if __name__ == "__main__":
    # 单元测试：验证 PDF转MD 全流程
    logger.info("===== 开始 node_pdf_to_md 节点单元测试 =====")
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    # 优先用doc 下的示例文件；没有则退化为「仅测 step_1」的参数校验
    sample_pdf = PROJECT_ROOT / "doc" / "hak180产品安全手册.pdf"

    if not sample_pdf.is_file():
        logger.warning(
            f"未找到测试用PDF：{sample_pdf}\n"
            f"请把任意 PDF 放到 {PROJECT_ROOT / 'doc'} 下后重试，"
            f"或改本文件里的 sample_pdf 指向你自己的 PDF。"
        )
        logger.info("===== 结束node_pdf_to_md 节点单元测试（跳过：缺少测试PDF）=====")
        raise SystemExit(0)

    test_state = {
        "task_id": "test_pdf2md_task_001",
        "pdf_path": str(sample_pdf),
        "local_dir": str(PROJECT_ROOT / "output"),
    }

    node_pdf_to_md(test_state)

    logger.info(f"md_path     = {test_state['md_path']}")
    logger.info(
        f"md_content  = {len(test_state['md_content'])} 字符"
    )
    logger.info("===== 结束 node_pdf_to_md 节点单元测试 =====")