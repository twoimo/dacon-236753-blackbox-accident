#!/usr/bin/env bash
# GPU 개발 기기 환경 구성 (conda 있으면 conda, 없으면 python venv 폴백).
# 평가 서버(docs/03 §6)와 라이브러리 핀을 맞춘다.
#
# 사용: bash env/scripts/setup_gpu.sh
# 근거: env/README.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="dacon236753"
PY_VER="3.11"
CU_INDEX="https://download.pytorch.org/whl/cu128"
REQ="env/configs/requirements-train.txt"

echo "[setup] repo: $REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  echo "[setup] conda 감지 → conda env '$ENV_NAME' 구성"
  if ! conda env list | grep -q "^${ENV_NAME}\b"; then
    conda create -y -n "$ENV_NAME" "python=${PY_VER}"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "[setup] conda 없음 → python venv 폴백 (.venv-gpu)"
  python${PY_VER} -m venv .venv-gpu 2>/dev/null || python3 -m venv .venv-gpu
  # shellcheck disable=SC1091
  source .venv-gpu/bin/activate
fi

python -m pip install --upgrade pip

echo "[setup] torch/torchvision (CUDA 12.8 휠)"
pip install --index-url "$CU_INDEX" torch==2.8.0 torchvision==0.23.0 || {
  echo "[setup] CUDA 휠 실패 → CPU 휠로 폴백 (개발용)"
  pip install torch==2.8.0 torchvision==0.23.0
}

echo "[setup] 나머지 의존성"
pip install -r "$REQ"

echo "[setup] 완료. 검증:"
python env/scripts/check_env.py
