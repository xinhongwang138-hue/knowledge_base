"""
掌柜智库知识库导入服务 —— 统一项目入口

本文件是项目的统一启动入口：只需 `python main.py` 即可拉起完整的
FastAPI Web 服务（复用 app/import_process/api/file_import_service.py 中
已装配好的 FastAPI 实例，含 CORS、路由、后台任务等全部能力）。

支持命令行参数：
    python main.py --host 0.0.0.0 --port 8010

约定沿用项目既有代码风格：
    1. 【必须放在最前】导入 app.core.torch_env，限制 MKL 线程数，
       早于任何 torch 初始化，否则 BGE-M3 向量化会 Segmentation fault；
    2. 日志统一走 app.core.logger 的 logger（loguru 单例）；
    3. uvicorn.run 的第一个参数用位置传递（不用 app=app），
       与 file_import_service.py 的启动写法保持一致。
"""
# 【必须放在最前】限制 MKL/OpenMP 线程数，早于任何 torch 初始化，
# 否则 BGE-M3 向量化会 Segmentation fault。原理见 app/core/torch_env.py。
# 导入即生效；下方启动日志会引用其初始化标志，便于确认环境变量已生效
import app.core.torch_env as torch_env

import argparse

import uvicorn

from app.core.logger import PROJECT_ROOT, logger
# 复用 Web 服务模块里已装配好的 FastAPI 实例（含 CORS、路由、后台任务等），
# 起别名 fastapi_app 避免与下方 argparse 等命名产生歧义
from app.import_process.api.file_import_service import app as fastapi_app


def parse_args() -> argparse.Namespace:
    """解析命令行参数：--host / --port"""
    parser = argparse.ArgumentParser(
        description="掌柜智库知识库导入服务统一入口（FastAPI）"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务监听地址，默认 127.0.0.1（仅本地访问），生产环境可设为 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="服务监听端口，默认 8010（8000 已被 Milvus Attu 占用）",
    )
    return parser.parse_args()


def main() -> None:
    """统一启动入口：解析参数 -> 拉起 FastAPI 服务"""
    args = parse_args()

    logger.info(f"MKL 线程限制环境已生效：{torch_env.TORCH_ENV_INITIALIZED}")
    logger.info(f"掌柜智库知识库导入服务启动中，项目根目录：{PROJECT_ROOT}")
    logger.info(f"监听地址：http://{args.host}:{args.port}，前端页面：http://{args.host}:{args.port}/import.html")

    # 第一个参数用位置传递（不用 app=app），与 file_import_service.py 保持一致
    uvicorn.run(fastapi_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
