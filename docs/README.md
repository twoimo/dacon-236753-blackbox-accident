# 문서 색인

| 문서 | 내용 | 성격 |
|---|---|---|
| [01-competition-overview.md](01-competition-overview.md) | 대회 배경·주제·3-Stage 정의·일정 | 대회 페이지 원문 보존 |
| [02-data-spec.md](02-data-spec.md) | Stage별 입출력 형식과 제출 CSV 규격 | 대회 페이지 원문 보존 |
| [03-evaluation-and-submission.md](03-evaluation-and-submission.md) | 채점식, `submit.zip` 구조, `inference.py` 인터페이스, 서버 제약 | 원문 + 상충 정리 |
| [04-datasets.md](04-datasets.md) | 로컬 레이아웃, CCD·comma2k19 상세, 라벨 생성법, 라이선스 | 실측 기반 |
| [05-data-integrity-report.md](05-data-integrity-report.md) | 무결성 점검 결과, 수정 내역, 남은 함정 | 실측 기반 |
| [06-collaboration.md](06-collaboration.md) | 워크트리·브랜치 운영, 데이터 공유 요령 | 운영 규칙 |
| [07-dataset-distribution.md](07-dataset-distribution.md) | 구글 드라이브 공개 링크 | **자동 생성** |

## 읽는 순서

- **처음 참여한다면** → 01 → 02 → 06 (셋업) → 05 (함정 파악)
- **모델을 만들려면** → 02 (입출력) → 03 (제출 규격) → 04 (학습 데이터)
- **데이터를 손보려면** → 05 → 04 → 06 (5절 생성물 규칙)

## 사실의 정본

수치나 스키마가 문서와 어긋나면 **[`catalog/catalog.json`](../catalog/catalog.json) 이 맞다.**
문서는 사람이 읽기 위한 요약이고, 카탈로그는 `scripts/build_catalog.py` 가 실제 파일을
스캔해 생성한다.

| 알고 싶은 것 | 볼 곳 |
|---|---|
| 대회 메타, Stage 스키마, 데이터셋 요약 | `catalog/catalog.json` |
| 파일 하나하나의 크기·sha256 | `catalog/files.csv` |
| 영상·이미지의 코덱·해상도·프레임수 | `catalog/media_index.csv` |
| CCD 이미지 75,000장 + 프레임 라벨 | `catalog/crashbest_index.csv` |
| CCD 영상 1,500개 집계 (첫 충돌 프레임 포함) | `catalog/crashbest_videos.csv` |
| 무결성 판정 | `catalog/integrity.json` |
| 배포 링크 | `catalog/distribution.json` |
