# 평가 서버(NVIDIA L40S, torch 2.8.0+cu128, docs/03 §5-6) 근사 개발 이미지.
# 목적: 팀원이 어느 기기에서든 평가 서버와 동일한 라이브러리로 학습/추론 재현.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 시스템 패키지 (docs/03 §6 일부)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git git-lfs \
        build-essential cmake ninja-build \
        libgl1 libglib2.0-0 \
        zip unzip p7zip-full jq curl zstd \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

WORKDIR /workspace

# 의존성 먼저 복사 → 레이어 캐시
COPY env/configs/requirements-train.txt /tmp/requirements-train.txt

# torch/torchvision은 CUDA 12.8 휠에서, 나머지는 PyPI에서
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 && \
    pip install -r /tmp/requirements-train.txt

# 코드는 런타임에 볼륨 마운트 (run_docker.sh)
CMD ["bash"]
