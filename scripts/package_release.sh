#!/usr/bin/env bash
#
# 깃허브에 올릴 수 없는 대용량 데이터를 배포용 아카이브로 묶는다.
#
#   bash scripts/package_release.sh            # dist/ 에 전체 아카이브 생성
#   bash scripts/package_release.sh samples    # 특정 번들만 생성
#
# 산출물은 dist/ 에 생성되고 dist/SHA256SUMS 로 검증할 수 있다.
# JPEG/HEVC 처럼 이미 압축된 데이터는 재압축 이득이 거의 없어 무압축 tar 로 묶고,
# 텍스트·CSV 위주 번들만 zstd 로 압축한다.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
cd "$ROOT"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m오류:\033[0m %s\n' "$*" >&2; exit 1; }

command -v zstd >/dev/null || die "zstd 가 필요합니다: brew install zstd"

mkdir -p "$DIST"

# tar 재현성 옵션: macOS bsdtar 확장속성/리소스포크 제외
TAR_COMMON=(--no-mac-metadata --exclude '.DS_Store' --exclude '__pycache__' --exclude '.ipynb_checkpoints')

bundle_samples() {
  # 내용이 거의 전부 h264/hevc 영상이라 재압축 이득이 없다. 무압축 tar.
  log "competition-samples: 대회 공개 예제 (stage1/2/3, 무압축 tar)"
  tar "${TAR_COMMON[@]}" -cf "$DIST/dacon236753-competition-samples.tar" \
    data/stage1 data/stage2 data/stage3
}

bundle_ccd() {
  # JPEG 75,000장. 무압축 tar.
  log "ccd-crashbest: CCD 크래시 프레임 75,000장 + 메타데이터 (무압축 tar)"
  tar "${TAR_COMMON[@]}" -cf "$DIST/dacon236753-ccd-crashbest.tar" \
    data/external/CrashBest data/external/Crash_Table.csv
}

bundle_comma2k19() {
  # HEVC 1개 + float64 센서 배열. 배열이 압축되므로 중간 레벨.
  log "comma2k19: 예제 세그먼트 + 유틸 (.git 제외, zstd -10)"
  tar "${TAR_COMMON[@]}" --exclude '.git' -cf - data/external/comma2k19 \
    | zstd -T0 -10 -o "$DIST/dacon236753-comma2k19.tar.zst" -f
}

bundle_catalog() {
  # 전부 CSV/JSON 텍스트. 압축률이 매우 높다.
  log "catalog: 기계판독 인덱스 (CSV/JSON, zstd -19)"
  tar "${TAR_COMMON[@]}" -cf - catalog \
    | zstd -T0 -19 -o "$DIST/dacon236753-catalog.tar.zst" -f
}

targets=("${@:-all}")
for target in "${targets[@]}"; do
  case "$target" in
    all)        bundle_samples; bundle_ccd; bundle_comma2k19; bundle_catalog ;;
    samples)    bundle_samples ;;
    ccd)        bundle_ccd ;;
    comma2k19)  bundle_comma2k19 ;;
    catalog)    bundle_catalog ;;
    *) die "알 수 없는 번들: $target (all|samples|ccd|comma2k19|catalog)" ;;
  esac
done

log "체크섬 생성"
( cd "$DIST" && shasum -a 256 *.tar *.tar.zst 2>/dev/null | sort -k2 > SHA256SUMS )

log "완료"
ls -lh "$DIST"
printf '\n총 용량: %s\n' "$(du -sh "$DIST" | cut -f1)"
