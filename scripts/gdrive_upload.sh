#!/usr/bin/env bash
#
# dist/ 아카이브를 구글 드라이브에 올리고, 링크만으로 누구나 내려받을 수 있는
# 공개 링크를 만들어 catalog/distribution.json 과 docs/07-dataset-distribution.md 에 기록한다.
#
#   bash scripts/gdrive_setup.sh      # 1회: OAuth 연결 (사람이 직접)
#   bash scripts/package_release.sh   # 아카이브 생성
#   bash scripts/gdrive_upload.sh     # 업로드 + 공개 링크 생성 (무인)
#
# 환경변수
#   RCLONE_REMOTE   기본 gdrive
#   DRIVE_DIR       기본 dacon-236753-blackbox-accident/v1
#   DRY_RUN=1       업로드 없이 계획만 출력

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_DIR="${DRIVE_DIR:-dacon-236753-blackbox-accident/v1}"
DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m오류:\033[0m %s\n' "$*" >&2; exit 1; }

command -v rclone >/dev/null || die "rclone 이 필요합니다: brew install rclone"
command -v python3 >/dev/null || die "python3 가 필요합니다."

rclone listremotes | grep -qx "${REMOTE}:" \
  || die "원격 '${REMOTE}' 이 없습니다. 먼저 'bash scripts/gdrive_setup.sh' 를 실행하세요."

[[ -d "$DIST" ]] || die "dist/ 가 없습니다. 먼저 'bash scripts/package_release.sh' 를 실행하세요."
mapfile -t ARCHIVES < <(cd "$DIST" && ls -1 *.tar *.tar.zst 2>/dev/null | sort)
(( ${#ARCHIVES[@]} )) || die "dist/ 에 아카이브가 없습니다."

log "업로드 대상 (${#ARCHIVES[@]}개, 총 $(du -sh "$DIST" | cut -f1))"
for name in "${ARCHIVES[@]}"; do
  printf '    %-46s %s\n' "$name" "$(du -h "$DIST/$name" | cut -f1)"
done
log "대상 경로: ${REMOTE}:${DRIVE_DIR}"

if [[ "$DRY_RUN" == "1" ]]; then
  warn "DRY_RUN=1 — 여기서 중단합니다."
  exit 0
fi

log "업로드 시작 (중단되면 같은 명령으로 재개 가능)"
rclone copy "$DIST" "${REMOTE}:${DRIVE_DIR}" \
  --progress \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 128M \
  --drive-acknowledge-abuse \
  --retries 5 \
  --low-level-retries 20 \
  --stats-one-line-per-file

log "공개 링크 생성 (링크가 있으면 로그인 없이 누구나 다운로드)"
FOLDER_LINK="$(rclone link "${REMOTE}:${DRIVE_DIR}" 2>/dev/null | tail -1 || true)"
[[ -n "$FOLDER_LINK" ]] || warn "폴더 링크 생성에 실패했습니다."

declare -a ENTRIES=()
for name in "${ARCHIVES[@]}" SHA256SUMS; do
  [[ -f "$DIST/$name" ]] || continue
  link="$(rclone link "${REMOTE}:${DRIVE_DIR}/${name}" 2>/dev/null | tail -1 || true)"
  if [[ -z "$link" ]]; then
    warn "링크 생성 실패: $name"
    continue
  fi
  file_id="$(printf '%s' "$link" | sed -n 's#.*/d/\([^/]*\).*#\1#p')"
  [[ -n "$file_id" ]] || file_id="$(printf '%s' "$link" | sed -n 's#.*[?&]id=\([^&]*\).*#\1#p')"
  bytes="$(stat -f%z "$DIST/$name" 2>/dev/null || stat -c%s "$DIST/$name")"
  sha="$(grep -E "  ${name}\$" "$DIST/SHA256SUMS" 2>/dev/null | awk '{print $1}' || true)"
  ENTRIES+=("${name}|${bytes}|${sha}|${link}|${file_id}")
  printf '    %-46s %s\n' "$name" "$link"
done

log "catalog/distribution.json 기록"
ROOT="$ROOT" FOLDER_LINK="$FOLDER_LINK" REMOTE="$REMOTE" DRIVE_DIR="$DRIVE_DIR" \
ENTRIES="$(printf '%s\n' "${ENTRIES[@]}")" python3 <<'PY'
import json, os, pathlib, datetime

root = pathlib.Path(os.environ["ROOT"])
entries = []
for line in os.environ["ENTRIES"].splitlines():
    if not line.strip():
        continue
    name, size, sha, link, file_id = line.split("|")
    entries.append({
        "file": name,
        "bytes": int(size),
        "sha256": sha,
        "share_link": link,
        "file_id": file_id,
        "direct_download": (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            if file_id else ""
        ),
    })

payload = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "provider": "google-drive",
    "access": "anyone-with-link (reader)",
    "rclone_remote": os.environ["REMOTE"],
    "drive_path": os.environ["DRIVE_DIR"],
    "folder_share_link": os.environ["FOLDER_LINK"],
    "total_bytes": sum(e["bytes"] for e in entries),
    "archives": entries,
    "verify": "shasum -a 256 -c SHA256SUMS",
    "restore": "bash scripts/fetch_data.sh",
}
out = root / "catalog" / "distribution.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# 데이터셋 배포 링크",
    "",
    "이 문서는 `scripts/gdrive_upload.sh` 가 자동 생성한다. 직접 수정하지 말 것.",
    "",
    f"- 생성: {payload['generated_at']}",
    f"- 접근 권한: {payload['access']} — 링크만 있으면 구글 로그인 없이 내려받을 수 있다.",
    f"- 총 용량: {payload['total_bytes'] / 1024**3:.2f} GiB",
    "",
    f"## 전체 폴더\n\n{payload['folder_share_link'] or '(생성 실패)'}",
    "",
    "## 개별 아카이브",
    "",
    "| 아카이브 | 용량 | 공유 링크 | 직접 다운로드 |",
    "|---|---|---|---|",
]
for e in entries:
    direct = f"[curl용]({e['direct_download']})" if e["direct_download"] else "-"
    lines.append(
        f"| `{e['file']}` | {e['bytes'] / 1024**2:.1f} MiB "
        f"| [열기]({e['share_link']}) | {direct} |"
    )
lines += [
    "",
    "## 복원",
    "",
    "```bash",
    "bash scripts/fetch_data.sh            # 전체 내려받기 + 압축 해제 + 검증",
    "bash scripts/fetch_data.sh samples    # 특정 번들만",
    "```",
    "",
    "무결성 검증은 `dist/SHA256SUMS` 와 `catalog/distribution.json` 의 `sha256` 값이 기준이다.",
    "",
]
(root / "docs" / "07-dataset-distribution.md").write_text("\n".join(lines), encoding="utf-8")
print(f"  catalog/distribution.json, docs/07-dataset-distribution.md 갱신 ({len(entries)}개 항목)")
PY

log "완료"
printf '\n폴더 공유 링크: %s\n' "${FOLDER_LINK:-(실패)}"
