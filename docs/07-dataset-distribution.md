# 데이터셋 배포 링크

> **아직 생성되지 않았다.** 이 문서는 `scripts/gdrive_upload.sh` 가 자동 생성한다.
> 직접 수정하지 말 것.

깃허브에 올릴 수 없는 대용량 데이터(약 8.3GB)를 구글 드라이브로 배포하고, 링크만
있으면 구글 로그인 없이 누구나 내려받을 수 있게 만드는 절차다.

## 생성 절차

```bash
bash scripts/gdrive_setup.sh      # 1회. 구글 OAuth 동의 — 사람이 직접 해야 한다.
make release                      # dist/ 에 배포 아카이브 생성
make publish                      # 업로드 + 공개 링크 생성 + 이 문서 자동 갱신
```

`gdrive_setup.sh` 만 수동이다. 브라우저에서 구글 계정 로그인과 접근 허용이 필요해
자동화할 수 없다. 한 번 끝내면 이후 업로드는 전부 무인으로 돌아간다.

생성이 끝나면 이 문서가 아래 내용으로 덮어써진다.

- 전체 폴더 공유 링크
- 아카이브별 공유 링크 + `curl` 용 직접 다운로드 URL
- 아카이브별 크기와 sha256

같은 정보가 [`catalog/distribution.json`](../catalog/distribution.json) 에 기계 판독
형태로도 기록된다.

## 아카이브 구성

| 번들 | 파일 | 내용 | 실측 크기 |
|---|---|---|---|
| `samples` | `dacon236753-competition-samples.tar` | `data/stage1`, `data/stage2`, `data/stage3` | 198 MiB |
| `ccd` | `dacon236753-ccd-crashbest.tar` | `data/external/CrashBest` (75,000장), `Crash_Table.csv` | ~8.0 GiB |
| `comma2k19` | `dacon236753-comma2k19.tar.zst` | `data/external/comma2k19` (`.git` 제외) | 56 MiB |
| `catalog` | `dacon236753-catalog.tar.zst` | `catalog/` 기계판독 인덱스 | 3.1 MiB |

압축 방식은 내용에 맞춰 다르다. JPEG·H.264·HEVC 는 이미 압축돼 있어 재압축 이득이
없으므로 `samples`·`ccd` 는 무압축 `tar` 다. `comma2k19` 는 float64 센서 배열이 섞여
있어 `zstd -10` (64.4 → 56.4 MiB), `catalog` 는 전부 CSV/JSON 텍스트라
`zstd -19` (11.8 → 3.1 MiB) 를 쓴다.

## 복원

```bash
make data                             # 전체
bash scripts/fetch_data.sh samples    # 일부만 (samples|ccd|comma2k19|catalog)
```

`fetch_data.sh` 는 sha256 을 대조하고, 이미 있고 해시가 맞는 파일은 다시 받지 않는다.
