#!/usr/bin/env bash
#
# rclone 구글 드라이브 원격을 1회 설정한다.
#
#   bash scripts/gdrive_setup.sh
#
# 이 단계는 구글 OAuth 동의 화면이 브라우저에 떠야 하므로 사람이 직접 해야 한다.
# 자동화할 수 없는 유일한 단계이며, 한 번만 하면 이후 업로드는 전부 무인 실행된다.

set -Eeuo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m주의:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m오류:\033[0m %s\n' "$*" >&2; exit 1; }

command -v rclone >/dev/null || die "rclone 이 필요합니다: brew install rclone"

if rclone listremotes | grep -qx "${REMOTE}:"; then
  log "원격 '${REMOTE}' 이 이미 설정되어 있습니다."
  rclone about "${REMOTE}:" || warn "원격 접근 확인에 실패했습니다. 토큰이 만료됐을 수 있습니다."
  exit 0
fi

cat <<'GUIDE'
──────────────────────────────────────────────────────────────────────
 rclone 구글 드라이브 연결 (1회, 약 2분)

 아래 대화형 설정이 열립니다. 순서대로 입력하세요.

   n                      새 원격 만들기
   gdrive                 원격 이름 (반드시 이 이름)
   drive                  스토리지 종류 검색 후 'drive' 선택
   (Enter)                client_id — 비워두면 rclone 공용 키 사용
   (Enter)                client_secret — 비워둠
   1                      scope: 전체 접근(Full access)
   (Enter)                service_account_file — 비워둠
   n                      고급 설정 사용 안 함
   y                      자동 설정(브라우저 인증) 사용
     -> 브라우저가 열리면 업로드할 구글 계정으로 로그인하고 접근을 허용
   n                      팀 드라이브(Shared Drive) 아님
   y                      설정 내용 확인
   q                      설정 종료

 client_id 를 비워두면 rclone 공용 API 키를 쓰게 되어 대용량 업로드 시
 속도 제한이 걸릴 수 있습니다. 8GB 이상을 올린다면 본인 구글 클라우드
 프로젝트에서 OAuth client ID 를 만들어 넣는 편이 훨씬 빠릅니다.
   https://rclone.org/drive/#making-your-own-client-id
──────────────────────────────────────────────────────────────────────
GUIDE

read -r -p "계속하려면 Enter, 취소하려면 Ctrl+C: " _

rclone config

rclone listremotes | grep -qx "${REMOTE}:" \
  || die "원격 '${REMOTE}' 이 만들어지지 않았습니다. 이름을 정확히 '${REMOTE}' 로 지정하세요."

log "연결 확인"
rclone about "${REMOTE}:"
log "완료. 이제 'bash scripts/gdrive_upload.sh' 를 실행하면 무인 업로드됩니다."
