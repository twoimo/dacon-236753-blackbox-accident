# 평가 및 제출

출처: [평가 페이지](https://dacon.io/competitions/official/236753/overview/evaluation) + [코드 제출 대회 일반 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)

> **주의 — 두 문서가 충돌한다.** 일반 가이드(대회 236564)는 진입점을 `script.py` 라고
> 설명하지만, 이 대회(236753)는 **`inference.py`** 를 요구하고 `script.py` 는 평가 서버가
> 자동 생성한다. 충돌 시 항상 이 대회의 평가 페이지 규칙이 우선한다.

---

## 1. 채점 방식

### Stage 1 — 재녹화 판별
원본/재녹화 두 클래스의 F1 을 동일 가중 평균하는 **Macro-F1**.

### Stage 2 — 사고 주요시점·상황
제출한 프레임 번호를 영상별 프레임–시간 대응정보로 **초 단위로 변환한 뒤** 정답 시각과 비교한다.

다음은 모두 오답 처리된다.

- 예측값 누락
- 결측값 또는 비수치값
- 음수 프레임
- 해당 영상 범위를 벗어난 프레임
- 허용된 범주 이외의 분류값

### Stage 3 — 차량 거동
조향 범주는 차량이 **주행 중인 유효 프레임만** 평가한다. `STOPPED` 프레임은 조향 평가에서
제외되지만, **제출 파일에는 전체 프레임의 `steer_label` 이 반드시 있어야 한다.**

### 종합 점수

| Stage | 가중치 |
|---|---|
| Stage 1 | 0.2 |
| Stage 2 | 0.4 |
| Stage 3 | 0.4 |

- 1차 평가: Private Score (대회 종료 시점의 Public Score)
- 2차 평가: 1차 상위 15팀 대상, '모델 개발 보고서' + '학습데이터 구성 보고서' 종합 평가

---

## 2. 제출물 구조

참가자가 만드는 `submit.zip`:

```text
submit.zip
├── model/
│   ├── stage1/
│   │   └── best.pt
│   ├── stage2/
│   │   ├── best.pt
│   │   └── resnet18-f37072fd.pth
│   └── stage3/
│       └── best.pt
├── inference.py        # 필수. 없으면 제출 횟수가 차감된다.
└── requirements.txt
```

평가 서버가 압축을 풀고 **자동으로 추가**하는 항목:

```text
├── data/               # 테스트 데이터 (읽기 전용, 쓰기·수정 불가)
├── script.py           # 채점용 진입점 (자동 생성)
└── output/
    └── submission.csv  # 여기에 결과가 저장되어야 한다
```

### `inference.py` 필수 인터페이스

```python
def predict_stage1(data_dir, model_dir):
    """반환: pandas.DataFrame[ID, answer]"""

def predict_stage2(data_dir, model_dir):
    """반환: pandas.DataFrame[ID, collision_frame, entry_frame, evasion_space, entry_side]"""

def predict_stage3(data_dir, model_dir):
    """반환: pandas.DataFrame[ID, sample_index, accel_label, steer_label]"""
```

`data_dir` 하위 레이아웃은 [02-data-spec.md](02-data-spec.md) 및
[05-data-integrity-report.md](05-data-integrity-report.md) 의 "평가 레이아웃" 절 참고.
Stage 2 만 이미지 폴더 입력이고 Stage 1·3 은 영상 파일 입력이라는 점이 실수하기 쉬운 지점이다.

---

## 3. 오프라인 환경 제약

평가 중 **인터넷이 차단**된다.

| 불가능 | 대안 |
|---|---|
| `mvit_v2_s(weights="DEFAULT")` 등 사전학습 가중치 자동 다운로드 | **`weights=None`** 으로 생성한 뒤 `model/` 에 넣어둔 `.pt` 를 직접 로드 |
| `AutoModel.from_pretrained("...")` (허브 조회) | 가중치를 `model/` 에 포함하고 로컬 경로로 로드 |
| 외부 API 호출, 원격 DB, 런타임 파일 다운로드 | 전부 `submit.zip` 안에 포함 |

`weights=None` 은 구버전 `pretrained=False` 에 해당한다. 베이스라인은
ResNet-18 ImageNet 가중치를 학습 시점에 받아 `model/stage2/resnet18-f37072fd.pth` 로
저장해 두고, 추론 시에는 그 파일을 읽는 방식을 쓴다.

---

## 4. 실행 제한

| 항목 | 제한 |
|---|---|
| `submit.zip` 용량 | 10GB 이내 |
| 압축 해제 후 용량 | 32GB 이내 |
| 패키지 설치 시간 | 10분 이내 |
| 전체 추론 시간 | 60분 이내 |

## 5. 평가 서버 사양

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA L40S, 44.7 GiB VRAM |
| CPU | 7 vCPU |
| RAM | 60GB (+ 공유 메모리 30GB) |
| 인터넷 | 차단 |

베이스라인 `inference.py` 는 CUDA 가 없으면 `RuntimeError` 를 던진다. 로컬 macOS 에서는
Stage 별 `predict_*` 를 그대로 실행할 수 없으므로, 전처리·후처리 로직만 분리해 검증하거나
`_device()` 를 우회하도록 수정해 시험해야 한다.

---

## 6. 평가 서버 기본 설치 패키지

`requirements.txt` 에 이미 호환되는 버전이 깔려 있는 패키지를 중복으로 넣으면 설치 시간만 잡아먹는다.

```text
torch==2.8.0+cu128, torchvision==0.23.0+cu128
pandas==2.2.2, numpy==1.26.4, scipy==1.15.3
scikit-learn==1.5.2, joblib==1.5.2, threadpoolctl==3.5.0
opencv-python-headless==4.10.0.84, pillow==10.4.0
av>=15,<17, imageio==2.37.0, imageio-ffmpeg==0.6.0
albumentations==1.4.20, scikit-image==0.24.0
timm==1.0.15, fvcore==0.1.5.post20221221
iopath==0.1.10, yacs==0.1.8, einops==0.8.1
transformers==4.57.6, accelerate==1.9.0
huggingface-hub==0.34.4, safetensors==0.6.2
sentencepiece==0.2.0, tokenizers>=0.22,<0.24
regex==2024.9.11, ultralytics==8.3.170
lap==0.5.12, filterpy==1.4.5, shapely==2.0.6
matplotlib==3.9.2, seaborn==0.13.2
tqdm==4.66.5, loguru==0.7.2, pyyaml==6.0.2
rich==13.9.4, psutil==6.1.1, omegaconf==2.3.0
hydra-core==1.3.2, torchmetrics>=1.4,<2
nvidia-ml-py>=12,<14
```

시스템 패키지 (일부): `ffmpeg`, `git`, `git-lfs`, `build-essential`, `cmake`, `ninja-build`,
`libavcodec-dev`, `libavformat-dev`, `libjpeg-dev`, `libpng-dev`, `libgl1`, `libglib2.0-0t64`,
`zip`, `unzip`, `p7zip-full`, `jq`

---

## 7. 오류 종류와 제출 횟수

| 종류 | 원인 | 제출 횟수 차감 |
|---|---|---|
| 설치 오류 | `submit.zip` 구조 불일치, 패키지 설치 실패 | 차감 안 됨 |
| 제출 오류 | `script.py` 실행 중 오류 | **차감됨** |

## 8. 제출 전 점검

- [ ] `submit.zip` 최상위에 `model/`, `inference.py`, `requirements.txt` 만 존재
- [ ] `predict_stage1` / `predict_stage2` / `predict_stage3` 세 함수 모두 정의
- [ ] 모든 모델 생성이 `weights=None`
- [ ] 사전학습 가중치 파일이 `model/` 안에 포함
- [ ] 온라인 의존 코드 전부 제거
- [ ] 반환 DataFrame 의 컬럼명·순서·범주값이 규격과 일치
- [ ] Stage 2 프레임 번호가 파일명 기준 원본 번호 (재번호 매기지 않음)
- [ ] Stage 3 `STOPPED` 프레임도 `steer_label` 포함
- [ ] 파일명에 공백·한글 없음
- [ ] 추론 시간 60분 이내

## 9. 문의

- 메일: dacon@dacon.io
- 대회 페이지 토크 탭
