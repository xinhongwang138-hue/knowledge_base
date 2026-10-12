# 知识库后端 API 镜像（导入服务 + 查询服务共用）
# 基于 Debian 的 Python 3.12 slim 镜像，安装项目全部依赖后拷贝代码。
# 模型权重不进镜像，运行时通过 bind-mount 挂载宿主机的 D:/ai_models 提供。
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# 系统级依赖：torch / opencv(magic-pdf) / 编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 torch 的 CPU 版本，避免后续依赖把庞大的 CUDA 版 torch 再拉一遍
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio

# 再装项目其余依赖（torch 已满足，不会被重复安装）
# 与 pyproject.toml 的 dependencies 保持一致
RUN pip install --no-cache-dir \
        "fastapi>=0.135.1" \
        "flagembedding>=1.3.5" \
        "grandalf>=0.8" \
        "langchain>=1.2.12" \
        "langchain-community>=0.4.1" \
        "langchain-openai>=1.1.11" \
        "langgraph>=1.1.2" \
        "loguru>=0.7.3" \
        "magic-pdf>=1.3.12" \
        "minio>=7.2.20" \
        "modelscope>=1.35.0" \
        "numpy>=2.4.3" \
        "pandas>=3.0.1" \
        "pymilvus[model]>=2.6.10" \
        "pymilvus-model>=0.3.2" \
        "pymongo>=4.16.0" \
        "python-dotenv>=1.2.2" \
        "python-multipart>=0.0.22" \
        "regex>=2026.2.28" \
        "uvicorn>=0.42.0" \
        "transformers>=4.57.6" \
        "openai-agents>=0.4.2" \
        "requests>=2.32.5"

# 拷贝项目代码（在依赖之后，源码改动不会让依赖层失效）
COPY . /app

# 模型缓存目录（指向挂载盘，避免重复下载）
ENV HF_HOME=/ai_models/hf_cache \
    MODELSCOPE_CACHE=/ai_models/modelscope_cache

EXPOSE 8010 8011
