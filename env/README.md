# env/ — 다기기 재현 실험 환경 (팀 온보딩)

여러 기기(로컬 macOS, CUDA 서버, 클라우드 GPU)에서 **동일한 환경**으로 실험하기 위한 세팅.
`autoresearch` 브랜치를 pull 한 팀원은 이 문서만 따라 하면 온보딩이 끝난다.

> **분리 원칙**: 이 브랜치는 자료조사·환경 공유 전용이다. 실제 모델 실험은 여기서 분기한
> **별도 실험 브랜치**에서 한다. 예: `git checkout -b exp/stage3-egomotion autoresearch`.

---

## 0. 3분 온보딩

```bash
git clone https://github.com/twoimo/dacon-236753-blackbox-accident.git
cd dacon-236753-blackbox-accident
git checkout autoresearch

# (A) 데이터/카탈로그만 볼 때 — GPU 불필요
make setup && make check          # .venv(python3.11)로 카탈로그 무결성 확인

# (B) 실제 학습/추론 — CUDA GPU 기기
bash env/scripts/setup_gpu.sh     # conda 또는 venv + torch cu128 스택
python env/scripts/check_env.py   # 환경 자기진단 (GPU/torch/ffmpeg/데이터)
```

## 1. 왜 두 갈래인가

| 상황 | 환경 | 이유 |
|---|---|---|
| 데이터 구조·카탈로그 확인, 문서 작업 | 루트 `.venv` (python3.11 + numpy + pillow) | 대회 베이스라인 추론이 CUDA 강제라 macOS에선 모델 실행 불가 |
| 실제 학습/추론 | CUDA GPU (Linux) + `env/` GPU 스택 | 평가 서버(L40S)와 동일 계열. `torch==2.8.0+cu128` |

로컬 macOS(M-시리즈)에서는 전처리/후처리/라벨 파이프라인만 검증하고, 학습·추론은 GPU 기기에서.
`env/scripts/check_env.py` 가 현재 기기가 어느 갈래인지 자동 판별한다.

## 2. 구성 요소

```
env/
├── README.md                 이 파일
├── configs/                  Stage별 실험 설정 (하이퍼파라미터·임계값·지표 계약)
│   ├── common.yaml               공통(디바이스/AMP/시간예산)
│   ├── stage1.yaml               재촬영 합성·재인코딩·검증 프로토콜
│   ├── stage2.yaml               충돌 국소화·약지도 라벨링
│   └── stage3.yaml               20→10Hz·조향 임계값 스윕
├── docker/
│   ├── Dockerfile                평가 서버와 유사한 CUDA 12.8 이미지
│   └── docker-compose.yml        GPU 컨테이너 + 데이터 볼륨 마운트
└── scripts/
    ├── setup_gpu.sh              GPU 기기 환경 구성 (conda/venv 자동 선택)
    ├── check_env.py              환경 자기진단 (GPU/torch/ffmpeg/데이터/카탈로그)
    └── run_experiment.sh         하니스 래퍼 (METRIC 로그 규약 강제)
```

## 3. 데이터

- 코드/카탈로그는 git에 있으나 대용량 데이터(`data/external/`, `data/stage3/videos/`)는 없다.
- 복원: `make data` (공개 구글드라이브 링크, `docs/07-dataset-distribution.md`).
- 여러 워크트리에서 8GB 데이터를 **한 벌만** 공유하는 법은 `docs/06-collaboration.md` 2절.

## 4. 실험 기록 규약 (autoresearch 방식)

- 모든 실험은 baseline/keep/discard 규율로 기록한다.
- 하니스는 성공 시 exit 0 + `METRIC <name>=<value>` 라인을 1개 이상 출력한다.
- 주 지표가 개선되면 `keep`, 나빠지거나 정체면 `discard`, 실패는 `crash`,
  검증 실패는 `checks_failed`.
- Stage별 주 지표는 `configs/*.yaml` 의 `primary_metric` 에 선언.

## 5. Docker (평가 서버 근사)

```bash
cd env/docker
docker compose build          # CUDA 12.8 + torch 2.8.0 + 대회 패키지 스택
docker compose run --rm gpu bash
# 컨테이너 안에서:
python env/scripts/check_env.py
```

인터넷 차단 환경을 모사하려면 `docker compose run --rm --network none gpu bash` 로 실행해
오프라인 추론(가중치 동봉) 가정을 검증한다.
