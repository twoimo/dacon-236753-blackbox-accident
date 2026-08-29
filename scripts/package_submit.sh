#!/usr/bin/env bash
# submit.zip 패키징 — model/ + inference.py + src/ + requirements.txt 를 담는다.
# 근거: docs/03-evaluation-and-submission.md §2, research/synthesis/.
#
# 사용: bash scripts/package_submit.sh [출력경로]
#   기본 출력: dist/submit.zip
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-dist/submit.zip}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "[package] 스테이징: $STAGE"

# 필수 구성물 (docs/03 §2)
mkdir -p "$STAGE/model"
[ -d model ] && cp -R model/. "$STAGE/model/" || echo "[package] 경고: model/ 없음 (가중치 미포함)"
cp inference.py "$STAGE/inference.py"
cp -R src "$STAGE/src"

# 제출용 requirements.txt: 서버 기설치본과 중복 최소화 (docs/03 §6).
# 실험에서 추가로 필요한 패키지만 여기에 적는다. 기본은 비움(=서버 스택 사용).
if [ -f submit_requirements.txt ]; then
  cp submit_requirements.txt "$STAGE/requirements.txt"
else
  : > "$STAGE/requirements.txt"   # 빈 파일 (서버 기설치 스택으로 충분한 경우)
fi

# 캐시/불필요 파일 제거
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
( cd "$STAGE" && zip -rq "$REPO_ROOT/$OUT" model inference.py src requirements.txt )

SIZE=$(du -h "$OUT" | cut -f1)
echo "[package] 생성 완료: $OUT ($SIZE)"
echo "[package] 내용:"; unzip -l "$OUT" | tail -n +2 | head -20

# 10GB 제한 경고
BYTES=$(wc -c < "$OUT")
if [ "$BYTES" -gt 10737418240 ]; then
  echo "[package] !! 경고: submit.zip 이 10GB 초과"
fi
echo "METRIC submit_zip_bytes=$BYTES"
