#!/usr/bin/env bash
# 실험 하니스 템플릿 (autoresearch 계약).
# 실험 브랜치에서 이 파일을 복사해 <RUN> 부분을 채운다.
#
# 계약:
#   - 성공 시 exit 0, 실패 시 non-zero
#   - 주 지표를 `METRIC <name>=<value>` 로 1줄 이상 출력
#   - 동일 워크로드 결정적 재현 (고정 시드, 네트워크 미사용)
#
# 사용: bash env/scripts/harness_template.sh env/configs/stage1.yaml
set -euo pipefail

CONFIG="${1:?사용: harness_template.sh <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[harness] config=$CONFIG"
START=$(date +%s)

# 1) 환경 정합 확인 (실패하면 checks_failed)
python env/scripts/check_env.py >/dev/null || { echo "checks_failed: env"; exit 3; }

# 2) <RUN> — 실험 브랜치에서 실제 학습/평가 커맨드로 교체
#    예: python -m src.stage1.evaluate --config "$CONFIG" --deterministic
echo "[harness] TODO: 실험 브랜치에서 실제 실행 커맨드로 교체하세요."
PRIMARY="nan"   # 실제 실행 결과에서 파싱

END=$(date +%s)
ELAPSED_MIN=$(awk "BEGIN{printf \"%.2f\", ($END-$START)/60}")

# 3) 지표 출력 (계약)
echo "METRIC primary=${PRIMARY}"
echo "METRIC infer_minutes=${ELAPSED_MIN}"
echo "ASI config=${CONFIG}"

# 4) 60분 예산 검증
python - "$ELAPSED_MIN" <<'PY'
import sys
elapsed = float(sys.argv[1])
if elapsed > 60:
    print(f"checks_failed: infer_minutes={elapsed} > 60")
    sys.exit(4)
PY
