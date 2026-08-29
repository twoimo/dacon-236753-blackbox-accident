# 블랙박스 영상 기반 지능형 고의사고 분석 (DACON 236753)

[대회 페이지](https://dacon.io/competitions/official/236753/overview/description) ·
상금 4,200만 원 · 리더보드 마감 2026-09-29

블랙박스 영상만을 입력으로 (1) 재녹화 여부, (2) 사고 주요시점·상황, (3) 차량 거동을
분석하는 모델을 만드는 **코드 제출** 대회다. 하나의 `submit.zip` 으로 세 Stage 결과를
모두 산출한다.

> **`autoresearch` 브랜치**: 근거 기반 문헌 리서치 + 다기기 재현 실험 환경은
> [`AUTORESEARCH.md`](AUTORESEARCH.md) 를 먼저 읽는다 (팀 온보딩 진입점).

| Stage | 과업 | 가중치 | 지표 |
|---|---|---|---|
| 1 | 원본 / 재녹화 이진 분류 | 0.2 | Macro-F1 |
| 2 | 충돌·진입 프레임 + 회피공간·진입방향 | 0.4 | 시각 오차 + 범주 정확도 |
| 3 | 0.1초 단위 가감속·조향 범주 | 0.4 | 범주 정확도 |

---

## 빠른 시작

```bash
make setup          # .venv 구성 (python3.11 + numpy + pillow)
make data           # 구글 드라이브에서 대용량 데이터 복원 (~8.3GB)
make stage2-images  # Stage2 평가 레이아웃 프레임 이미지 생성
make check          # 무결성 검증
make                # 전체 명령 목록
```

필요한 도구: `git`, `python3.11+`, `ffmpeg`, `zstd`, `curl`.
데이터 복원에 구글 계정이나 `rclone` 설정은 필요 없다 — 공개 링크로만 동작한다.

여러 사람이 함께 작업한다면 [docs/06-collaboration.md](docs/06-collaboration.md) 를
먼저 읽는다. 특히 워크트리 여러 개에서 8GB 데이터를 한 벌만 두고 공유하는 방법이 2절에 있다.

---

## 구조

```text
.
├── README.md                  이 파일
├── AGENTS.md                  AI 에이전트용 작업 규칙
├── Makefile                   모든 워크플로 진입점
├── docs/                      사람이 읽는 문서 (docs/README.md 가 색인)
├── catalog/                   ★ 기계 판독 정본 — 데이터 없이도 조회 가능
│   ├── catalog.json               대회 메타 + Stage 스키마 + 데이터셋 요약
│   ├── files.csv                  파일별 크기·sha256
│   ├── media_index.csv            영상·이미지 코덱·해상도·프레임수
│   ├── crashbest_index.csv        CCD 75,000장 × 프레임 라벨 조인
│   ├── crashbest_videos.csv       CCD 1,500 영상 집계
│   ├── integrity.json             무결성 판정
│   └── distribution.json          배포 링크 (publish 후 생성)
├── scripts/                   카탈로그 생성·검증·라벨 파생·배포
├── research/                  ★ autoresearch 브랜치 — 근거 기반 문헌 리뷰·전략
│   ├── references/                논문 신빙성 정본 (학회/저널 등급 + 인용수)
│   ├── synthesis/                 종합 전략 + autoresearch verdict
│   └── 0X-*/                      Stage별 상세 (재녹화·사고시점·거동·백본)
├── env/                       ★ 다기기 재현 실험 환경 (팀 온보딩)
│   ├── configs/                   Stage별 실험 설정 (하이퍼파라미터·지표 계약)
│   ├── docker/                    평가 서버 근사 CUDA 이미지
│   └── scripts/                   환경 세팅·자기진단·하니스 템플릿
├── baseline/                  대회 배포 베이스라인 (원본 보존, 미수정)
└── data/                      평가 서버와 동일한 레이아웃
    ├── stage1/  labels.csv + videos/{original,rerecorded}/
    ├── stage2/  labels.csv + videos/ + images/<ID>/
    ├── stage3/  videos/ + labels_comma2k19.csv
    └── external/  CrashBest/ · Crash_Table.csv · comma2k19/
```

`data/` 는 평가 서버가 `data_dir` 아래에 구성하는 구조를 그대로 미러링한다. 덕분에
`baseline/baseline_train.ipynb` 의 `DATA = ROOT/'data'` 가 수정 없이 동작한다.

### 깃에 없는 것

| 경로 | 용량 | 복원 |
|---|---|---|
| `data/external/` | 8.1GB | `make data` |
| `data/stage3/videos/` | 179MB | `make data` |
| `data/stage2/images/` | 12MB | `make stage2-images` |

링크 목록은 [docs/07-dataset-distribution.md](docs/07-dataset-distribution.md).

---

## 카탈로그 — AI/LLM 색인용 정본

데이터셋에 대한 모든 사실은 `catalog/` 한 곳에서 나온다.
`scripts/build_catalog.py` 가 실제 파일을 스캔해 생성하므로 문서와 어긋날 일이 없다.
**8GB 데이터를 내려받지 않아도** 저장소만 클론하면 전체 데이터셋을 조회할 수 있다.

```bash
# Stage 2 정답 좌표계 확인
jq '.stages.stage2' catalog/catalog.json

# 야간 + 눈 + 자차 관여 크래시 영상 찾기
awk -F, '$5=="Night" && $6=="Snowy" && $7=="Yes" {print $1, $3}' catalog/crashbest_videos.csv

# 충돌이 20프레임 이전에 시작하는 영상
awk -F, 'NR>1 && $3!="" && $3<20 {print $1, $3}' catalog/crashbest_videos.csv

# 프레임 단위 조회: 000001 영상의 충돌 프레임만
awk -F, '$2=="000001" && $5==1 {print $1}' catalog/crashbest_index.csv

# 무결성 상태
jq '.summary, [.findings[] | select(.severity=="error")]' catalog/integrity.json
```

CCD 1,500 영상의 **첫 충돌 프레임**(= Stage 2 `t_collision` 정의)은
`crashbest_videos.csv` 의 `first_crash_frame_index` 컬럼에 0-기반으로 미리 계산돼 있다.

---

## 데이터 상태 — 읽고 시작할 것

전체 점검 결과는 [docs/05-data-integrity-report.md](docs/05-data-integrity-report.md) 에 있다.
모델을 만들기 전에 알아야 할 세 가지만 옮긴다.

**1. 공개 예제만으로 Stage 1 을 학습하면 안 된다.**
배포된 ORIGINAL 5건은 100% mpeg4, RERECORDED 5건은 100% h264 다. 컨테이너
메타데이터만 읽어도 라벨이 맞는다. 게다가 재녹화 예제는 실제 재촬영본이 아니라
리샘플링·노이즈를 모사한 파생본이다. CCD 75,000장을 원본 풀로 삼아 재녹화를 직접
합성하고, 두 클래스를 **같은 코덱·비트레이트로 재인코딩**해 코덱 단서를 없애야 한다.

**2. Stage 3 공개 예제는 20Hz 인데 대회 스펙은 10Hz 다.**
`data/stage3/videos/*.mp4` 는 comma2k19 원본 그대로 20Hz(패킷 간격 0.05초)이고,
컨테이너는 40fps 로 잘못 선언되어 있으며 후반부 PTS 가 손상돼 `duration` 을 믿을 수 없다.
`sample_index` 를 프레임마다 발급하면 시간축이 2배로 어긋난다.

**3. Stage 3 정답 라벨은 로컬에 없고 복구할 수 없다.**
대체로 comma2k19 CAN 정답에서 파생한 `data/stage3/labels_comma2k19.csv` (600행)를
제공한다. 전체 comma2k19(~100GB)를 받으면 같은 스크립트로 약 120만 행까지 늘릴 수 있다.

수정 완료된 항목: Stage 1 의 ORIGINAL 클래스가 RERECORDED 복사본으로 덮여 있던 문제
(복구), Stage 2 라벨 소실 (CCD 공식 주석에서 재생성), 로컬 경로가 베이스라인·평가
서버 레이아웃과 어긋나던 문제 (`data/` 로 통일).

---

## 학습 데이터

대회는 **학습 데이터셋을 제공하지 않는다.** 배포되는 15개 영상은 입출력 형식 확인용이고,
참가자가 법적 제한 없는 데이터를 직접 구성해 2차 평가에서 그 내역을 보고해야 한다.

| 데이터셋 | 규모 | 쓰임 |
|---|---|---|
| CCD (Car Crash Dataset) | 75,000 프레임 / 1,500 크래시 영상 + 프레임별 충돌 주석 | Stage 1 원본 풀, Stage 2 시점 국소화 |
| comma2k19 | 예제 세그먼트 1개 (CAN·IMU·GNSS·pose) | Stage 3 CAN 정답 기반 라벨 |
| 대회 공개 예제 | 15 영상 | 코드 실행 확인 전용 |

상세와 확보 경로는 [docs/04-datasets.md](docs/04-datasets.md).

---

## 베이스라인

`baseline/` 은 대회 배포본을 **수정 없이** 보존한다.

| Stage | 백본 | 입력 |
|---|---|---|
| 1 | MViT V2 S | 16프레임 × 224×224 → 2-class |
| 2 | ResNet-18 → BiGRU(512→192, 2층) | 프레임 시퀀스 → 충돌·진입 시점 + 장면 4-class |
| 3 | MViT V2 S | 16프레임 × 224×224 → 가감속 4-class + 조향 3-class |

평가 서버는 인터넷이 차단되므로 모든 모델을 **`weights=None`** 으로 만들고 가중치
파일을 `model/` 에 넣어야 한다. 자세한 제약은
[docs/03-evaluation-and-submission.md](docs/03-evaluation-and-submission.md).

> `baseline_inference.ipynb` 의 `inference.py` 생성 셀은 노트북 파일명을 하드코딩해서
> 로컬에서는 `FileNotFoundError` 가 난다. 원본 보존을 위해 고치지 않았다.
> 자세한 내용은 [docs/05](docs/05-data-integrity-report.md) 10절.

---

## 명령 참조

| 명령 | 하는 일 |
|---|---|
| `make setup` | `.venv` 구성 |
| `make data` | 구글 드라이브에서 대용량 데이터 복원 |
| `make labels` | 라벨 재생성 (Stage 2 복원 + comma2k19 파생) |
| `make stage2-images` | Stage 2 평가 레이아웃 프레임 이미지 생성 |
| `make catalog` | 카탈로그 + 무결성 판정 재생성 (sha256 포함, ~1분) |
| `make catalog-fast` | 같은 작업, sha256 생략 |
| `make check` | 카탈로그 대조 + 무결성 요약 |
| `make verify-full` | sha256 전수 대조 |
| `make release` | 배포 아카이브 생성 (`dist/`) |
| `make publish` | 구글 드라이브 업로드 + 공개 링크 생성 |
| `make clean-derived` | 재생성 가능한 산출물 삭제 |
