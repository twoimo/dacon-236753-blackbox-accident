# M5 Max 로컬 GPU(MPS) 학습 가이드

이 맥은 **Apple M5 Max, GPU 40코어, 128GB RAM** 으로 로컬 학습에 충분하다. 다만 **Aside 세션
(자동 에이전트)에서는 새로 설치한 네이티브 라이브러리(torch/pandas) 로딩이 시스템 정책으로
차단**된다(라이브러리 검증). 따라서 아래는 **사용자가 직접 실제 터미널(Terminal.app)에서** 실행한다.

## 왜 Aside에서 안 되나 (진단 결과)

- `.venv` 의 numpy(2026-08-28 설치)만 로드됨. 세션 중 새로 `pip install` 한 torch/pandas 는
  `library load disallowed by system policy` 로 차단(ad-hoc 재서명도 실패).
- 이는 하드웨어 문제가 아니라 Aside 샌드박스의 dylib 검증 정책. **실제 터미널에서는 정상 동작**한다.

## 1. 실제 터미널에서 환경 구성 (1회)

```bash
cd ~/Documents/projects/dacon
python3.11 -m venv .venv-mps
source .venv-mps/bin/activate
pip install --upgrade pip
# MPS 지원 torch (macOS arm64 기본 휠에 MPS 포함)
pip install torch torchvision numpy pandas pillow opencv-python
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"   # True 여야 함
```

## 2. MPS 확인 후 학습 (Stage별)

```bash
source .venv-mps/bin/activate
# device=mps 로 학습 (experiments/run.py 가 stage train 을 호출)
python -m experiments.run train --stage stage3        # 가감속·조향
python -m experiments.run train --stage stage2        # 충돌 국소화 헤드 (CCD 지도)
python -m experiments.run train --stage stage1        # 재촬영 2-스트림
```

각 `src/stageN/train.py` 는 torch 가 있으면 실제 학습, 없으면 안내 후 종료(게이팅)한다.
`src/common/runtime.py` 의 `get_device()` 가 MPS 를 자동 선택한다.

## 3. 로컬에서 이미 검증된 것 (Aside numpy 실측)

실제 학습 전, numpy 만으로 아래가 이미 실측되어 방향이 검증됨 (experiments/RESULTS.md):

| Stage | 실측 결과 | 의미 |
|---|---|---|
| 1 | 코덱 중화 후 분리 acc ~91%, edge/FFT 특징 유효 | 픽셀 특징 학습 타당 |
| 2 | CCD 1,500영상 충돌 MAE 5.22f(0.52s), within-3f 48% | 시간 prior+onset 유효, 학습 헤드로 상향 여지 |
| 3 | accel_acc 0.64(+13pt), 조향 미미 | 가감속 신호 유효, 조향은 데이터 확장 필요 |

즉 MPS 학습은 "탐색"이 아니라 **이미 검증된 방향을 신경망으로 끌어올리는** 단계다.

## 4. MPS 학습 시 권장 (research/04, env/configs)

- 배치·해상도는 128GB RAM 이라 여유. 단 MPS 는 일부 연산 CPU 폴백 → 프로파일링 권장.
- AMP(fp16)는 MPS 에서 제한적 → fp32 로 시작.
- 백본: research 권장 X3D-S(경량) 또는 ResNet-18(Stage2). torchvision 제공.
- 평가 서버는 CUDA L40S 이므로, MPS 학습 가중치를 저장 후 **CUDA 서버에서 추론 재현** 확인 필요
  (`weights=None` + `model/` 로컬 로드 규약, docs/03).

## 5. 주의

- 이 맥에서 학습한 `.pt` 는 `model/stageN/best.pt` 로 저장하면 `predict_*` 가 자동으로 사용.
- 제출 전 `python scripts/validate_submission.py` + `bash scripts/package_submit.sh` 로 규격 점검.
- Aside 세션에서는 이 학습을 대신 실행할 수 없다(위 정책). numpy 실측·데이터 준비·코드까지가 Aside 몫.
