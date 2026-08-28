#!/usr/bin/env bash
#
# catalog/distribution.json 의 공개 링크에서 대용량 데이터를 내려받아 복원한다.
# 저장소를 새로 클론한 사람이 데이터를 갖추는 표준 경로다. 구글 계정이나
# rclone 설정이 필요 없다 — 공개 링크만으로 동작한다.
#
#   bash scripts/fetch_data.sh                    # 전체
#   bash scripts/fetch_data.sh samples catalog    # 일부만
#   KEEP_ARCHIVES=1 bash scripts/fetch_data.sh    # 내려받은 아카이브 보존
#
# 번들 이름: samples | ccd | comma2k19 | catalog

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
MANIFEST="$ROOT/catalog/distribution.json"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
cd "$ROOT"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m오류:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl 이 필요합니다."
command -v python3 >/dev/null || die "python3 가 필요합니다."
command -v zstd >/dev/null || warn "zstd 가 없으면 .tar.zst 를 풀 수 없습니다: brew install zstd"

[[ -f "$MANIFEST" ]] || die "$MANIFEST 가 없습니다.
데이터가 아직 업로드되지 않았습니다. 데이터를 가진 사람이
  bash scripts/package_release.sh && bash scripts/gdrive_upload.sh
를 실행해 링크를 생성한 뒤 커밋해야 합니다."

bundle_pattern() {
  case "$1" in
    samples)   echo "competition-samples" ;;
    ccd)       echo "ccd-crashbest" ;;
    comma2k19) echo "comma2k19" ;;
    catalog)   echo "catalog" ;;
    *) die "알 수 없는 번들: $1 (samples|ccd|comma2k19|catalog)" ;;
  esac
}

# 인자가 없으면 전체. 있으면 번들 이름을 파일명 패턴으로 바꾼다.
# 빈 문자열을 sentinel 로 쓰면 명령치환이 개행을 지워 파이썬 쪽에서 빈 리스트가
# 되어 "전체"가 "아무것도"로 뒤집힌다. 그래서 명시적으로 ALL 을 쓴다.
if (( $# == 0 )); then
  patterns=("ALL")
else
  patterns=()
  for name in "$@"; do patterns+=("$(bundle_pattern "$name")"); done
fi

mkdir -p "$DIST"

# 매니페스트에서 (파일명, sha256, 다운로드URL) 목록을 뽑는다.
mapfile -t ROWS < <(
  MANIFEST="$MANIFEST" PATTERNS="$(printf '%s\n' "${patterns[@]}")" python3 <<'PY'
import json, os, sys
manifest = json.load(open(os.environ["MANIFEST"], encoding="utf-8"))
patterns = [p for p in os.environ["PATTERNS"].splitlines() if p]
take_all = "ALL" in patterns
for archive in manifest.get("archives", []):
    name = archive["file"]
    if name == "SHA256SUMS":
        continue
    if not take_all and not any(p in name for p in patterns):
        continue
    url = archive.get("direct_download") or archive.get("share_link", "")
    if not url:
        print(f"경고: {name} 에 다운로드 URL 이 없습니다.", file=sys.stderr)
        continue
    print(f"{name}\t{archive.get('sha256','')}\t{url}")
PY
)

(( ${#ROWS[@]} )) || die "매니페스트에서 대상 아카이브를 찾지 못했습니다."

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r name sha url <<<"$row"
  target="$DIST/$name"

  if [[ -f "$target" && -n "$sha" ]] && shasum -a 256 "$target" | awk '{print $1}' | grep -qx "$sha"; then
    log "$name — 이미 있고 해시 일치, 다운로드 생략"
  else
    log "$name 다운로드"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$target" "$url" \
      || die "다운로드 실패: $name
브라우저에서 직접 받으려면 docs/07-dataset-distribution.md 의 링크를 사용하세요."
    if [[ -n "$sha" ]]; then
      actual="$(shasum -a 256 "$target" | awk '{print $1}')"
      [[ "$actual" == "$sha" ]] || die "해시 불일치: $name
  기대 $sha
  실제 $actual
구글 드라이브가 대용량 파일에 확인 페이지(HTML)를 반환했을 수 있습니다.
docs/07-dataset-distribution.md 의 공유 링크로 브라우저에서 직접 받아
$DIST 에 두고 이 스크립트를 다시 실행하세요."
      log "  해시 확인 OK"
    fi
  fi

  log "$name 압축 해제"
  case "$name" in
    *.tar.zst) zstd -dc "$target" | tar -xf - -C "$ROOT" ;;
    *.tar)     tar -xf "$target" -C "$ROOT" ;;
    *) warn "알 수 없는 형식, 건너뜀: $name"; continue ;;
  esac
  [[ "$KEEP_ARCHIVES" == "1" ]] || rm -f "$target"
done

log "복원 완료. 무결성 검증:"
printf '    python scripts/verify_integrity.py --hash --sample 2000\n\n'
