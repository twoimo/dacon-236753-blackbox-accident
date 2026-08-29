#!/usr/bin/env bash
# 평가 서버 근사 CUDA 컨테이너 실행. 프로젝트 루트를 마운트하고 GPU를 전달한다.
#
# 사용:
#   bash env/scripts/run_docker.sh            # 대화형 bash
#   bash env/scripts/run_docker.sh offline    # 인터넷 차단 모사 (--network none)
# 근거: env/README.md, docs/03 (오프라인 평가)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="dacon236753:dev"
MODE="${1:-interactive}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[run_docker] 이미지 빌드: $IMAGE"
  docker build -f "$REPO_ROOT/env/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
fi

NET_ARGS=()
if [[ "$MODE" == "offline" ]]; then
  echo "[run_docker] 오프라인 모드 (--network none) — 가중치 동봉 가정 검증"
  NET_ARGS=(--network none)
fi

GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
else
  echo "[run_docker] NVIDIA 런타임 미감지 → CPU 컨테이너 (전처리/후처리 검증용)"
fi

exec docker run --rm -it \
  "${GPU_ARGS[@]}" "${NET_ARGS[@]}" \
  --shm-size=30g \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  "$IMAGE" bash
