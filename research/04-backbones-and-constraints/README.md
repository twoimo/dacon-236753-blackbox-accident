# Backbones & 하드웨어 제약 (Stage 공통)

- **평가 서버**: NVIDIA **L40S 44.7GiB**, 7 vCPU, 60GB RAM(+30GB shm), **인터넷 차단**.
- **실행 제약**: 추론 **60분**, 패키지 설치 10분, submit.zip 10GB / 해제 후 32GB.
- **근거 정본**: `research/references/README.md`.

---

## 1. 60분 추론 예산이 백본 선택을 지배한다

3개 Stage를 **하나의 코드로** 60분 안에 전부 추론해야 한다. 따라서 정확도-지연 트레이드오프가
아키텍처 1순위 기준이다. 검증된 후보(모두 최상위 학회 + 고인용):

| 백본 | 게재처 | 특징 | Stage 적합 |
|---|---|---|---|
| **X3D** (`X3D2020`, CVPR20, ≈1,882 cites) | 초경량 3D CNN, mobile-regime | 60분 제약에 최적 | 1,2,3 전반 |
| **MViTv2** (`MVITV2-2022`, CVPR22) | torchvision `mvit_v2_s`, 베이스라인 백본 | 정확도 높음, 비용 중간 | 1,3 (베이스라인 호환) |
| **MViT v1** (`MVIT2021`, ICCV21) | 계층적 풀링 어텐션, 외부 사전학습 불필요 | 데이터 적을 때 강함 | 1,3 |
| **Video Swin** (`VSWIN2021`) | 지역성 귀납편향, 속도-정확도 우수 | 중형 | 2,3 |
| **MoViNets** (`MOVINET2021`, CVPR21) | Stream Buffer로 메모리 상수화 | 긴 영상 온라인 | 3 (긴 주행 클립) |

**전략**: 베이스라인은 `mvit_v2_s` 를 쓴다. 우리는 **X3D-S/-M 로 경량화한 변형**을 병행 검토해
60분 예산 여유를 확보하고, 남는 예산을 TTA/앙상블에 재투자하는 방향을 실험한다.

## 2. 인터넷 차단 — 사전학습 가중치 처리 (docs/03 반영)

- `mvit_v2_s(weights="DEFAULT")` 같은 **자동 다운로드 금지**. 반드시 `weights=None` 으로 생성 후
  `model/` 에 동봉한 `.pt` 를 **로컬 경로로 로드**.
- HuggingFace `from_pretrained("...")` (허브 조회) 금지. 가중치를 zip에 포함.
- 베이스라인 관행: ResNet-18 ImageNet 가중치를 **학습 시점에** 받아
  `model/stage2/resnet18-f37072fd.pth` 로 저장 → 추론 때 그 파일만 읽음.

## 3. 패키지 — 서버 기본 설치본과 중복 최소화 (docs/03 §6)

서버에 이미 깔린 버전(중복 넣으면 설치시간만 소모):
- `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, `timm==1.0.15`,
  `opencv-python-headless==4.10.0.84`, `av`, `ultralytics==8.3.170`, `albumentations`, 등.
- `requirements.txt` 에는 **서버에 없는 것만** 추가. 10분 설치 제약 방어.

## 4. VRAM 44.7GB 활용

- 44.7GB는 넉넉한 편 → 배치·해상도·클립 길이를 키울 여지. 단 **추론 시간이 병목**이므로
  VRAM을 정확도로 바꾸되 60분을 넘기지 않게 프로파일링한다.
- 혼합정밀(AMP/fp16), `torch.compile`, `channels_last`, 배치 추론으로 처리량 확보.

## 5. 제출 구조 (docs/03 §2, 자주 틀리는 지점)
```
submit.zip
├── model/stage1/best.pt, stage2/{best.pt, resnet18-f37072fd.pth}, stage3/best.pt
├── inference.py   # predict_stage1/2/3 필수
└── requirements.txt
```
- **Stage 2만 이미지 폴더 입력**, Stage 1·3은 영상 파일 입력. 입력 로더에서 자주 틀림.
- 파일명 공백·한글 금지. Stage 3 STOPPED 프레임도 `steer_label` 포함.

## 6. 로컬(macOS) 개발 주의
- 베이스라인 추론은 CUDA 강제 → macOS에서 `predict_*` 직접 실행 불가.
- 로컬에선 전처리/후처리 로직만 분리 검증하거나 `_device()` 우회. 실제 학습/추론은
  CUDA GPU 기기(팀 공용 서버/클라우드)에서. `env/` 가 그 이식성을 담당.

## 7. 검증 가능한 실험 후크
- `env/configs/common.yaml`: `device`, `amp`, `channels_last`, `torch_compile`, `time_budget_min`.
- 하니스 지표: `METRIC infer_minutes=<value>` (3-Stage 합산 추론 시간, 60 미만 필수) +
  Stage별 정확도 지표. 시간 초과 run은 `checks_failed` 로 기록.
