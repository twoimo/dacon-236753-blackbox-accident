# EXPERIMENTS — 실험 브랜치 온보딩 (exp/*)

> DACON 236753 리더보드 1등을 위한 **실제 실험** 브랜치. `autoresearch` 브랜치(자료·전략·환경)에서
> 분기해, 여기서 모델을 학습하고 제출물을 만든다. 근거는 `research/`, 환경 계약은 `env/`.

## 0. 이 브랜치의 위치

```
main ──┬── autoresearch      # 자료조사·전략·재현환경 (코드 없음)
       └── exp/baseline      # ← 지금 여기. 실행 가능한 실험 뼈대 + 제출 파이프라인
              └── exp/<주제>  # 각자 실험은 여기서 다시 분기 권장
```

- `autoresearch` 는 findings/verdict 전용. **모델 코드는 exp/* 에서만** 작성한다.
- 새 실험은 `git checkout -b exp/stage3-flow exp/baseline` 처럼 분기한다.

## 1. 구조 (이 브랜치에서 추가된 것)

```text
inference.py                  ★ 제출 진입점. predict_stage{1,2,3} 를 노출 (평가 서버가 호출)
src/
├── common/
│   ├── config.py             env/configs/*.yaml 로더 (common+stage 병합)
│   ├── data.py               평가 서버 레이아웃과 1:1 경로/로딩 유틸
│   ├── runtime.py            시드/디바이스/AMP + ffprobe 프레임 실측 + 20→10Hz 리샘플
│   └── submit_guard.py       제출 규격 방어 (범주 화이트리스트/정수화/범위 clamp)
├── stage1/  predict.py · train.py    재녹화 판별
├── stage2/  predict.py · train.py    사고 시점·상황
└── stage3/  predict.py · train.py    가감속·조향
experiments/run.py            실험 CLI (train/predict 를 METRIC 계약으로 래핑)
scripts/
├── validate_submission.py    제출 전 규격 자동 점검 (docs/03 §8)
└── package_submit.sh         submit.zip 패키징 (model/ + inference.py + src/ + requirements.txt)
```

## 2. 온보딩 (GPU 기기)

```bash
git checkout exp/baseline
bash env/scripts/setup_gpu.sh        # torch 2.8.0+cu128 + 의존성 (env/ 참고)
python env/scripts/check_env.py      # METRIC env_ok=1 확인

# 데이터 없으면 복원 (docs/07)
make data && make labels && make stage2-images
```

macOS(Apple Silicon)에서는 CUDA가 없어 학습/추론 불가. 아래 "로컬에서 검증 가능한 것"만 돌린다.

## 3. 실행 방법

### 학습 (Stage별)
```bash
python -m experiments.run train --stage stage1 --dry-run   # 설정 계약만 점검
python -m experiments.run train --stage stage1             # 실제 학습 (GPU, 구현 후)
```

### 추론 (제출 함수 직접 호출)
```bash
python -m experiments.run predict --stage stage3 --data-dir data/stage3 --model-dir model/stage3
```

### 제출물 만들기
```bash
python scripts/validate_submission.py          # repo 상태 사전 점검
bash scripts/package_submit.sh dist/submit.zip # 패키징
python scripts/validate_submission.py dist/submit.zip   # zip 규격 점검
```

## 4. 실험 기록 규약 (autoresearch 하니스 계약)

- 모든 실행은 성공 시 **exit 0** + `METRIC <name>=<value>` 를 1줄 이상 출력.
- 주 지표 개선 → `keep`, 정체/악화 → `discard`, 실패 → `crash`, 규격 위반 → `checks_failed`.
- Stage별 주 지표(`env/configs/*.yaml` 의 `primary_metric`):
  - Stage1 `macro_f1` (진짜 지표는 `lodo_macro_f1`, 높을수록 좋음)
  - Stage2 `collision_mae_sec` (낮을수록 좋음)
  - Stage3 `mean_acc` = (accel_acc + steer_acc_moving)/2 (높을수록 좋음)
- **추론 60분 예산**: 3-Stage 합산. 초과 시 `checks_failed` 로 기록하고 경량화.

## 5. 지금 상태 (스캐폴드 → 실제 구현 전환)

`predict_*` 는 **실행 가능한 스캐폴드**다: 경로 처리·제출 규격·프레임률 정합은 이미 올바르고,
모델이 없으면 안전 기본값을 반환한다. `model/stage{N}/best.pt` 가 있으면 실제 추론을 하도록
`TODO(exp)` 지점을 채우면 된다.

**우선순위 (research/synthesis 근거, 가중치順):**
1. **Stage 3** (0.4): `train.py` 에 comma2k19 라벨 로드 → 옵티컬플로우/ego-motion 특징 → accel/steer 분류.
   조향 임계값 스윕(`env/configs/stage3.yaml labels.sweep`) 먼저.
2. **Stage 2** (0.4): CCD 첫 positive frame(`catalog/crashbest_videos.csv`)로 충돌 국소화 지도학습.
3. **Stage 1** (0.2): CCD 프레임에서 재촬영 합성 → **두 클래스 동일 재인코딩**(코덱 누설 차단) → LODO 검증.

## 6. 반드시 지킬 함정 (코드에 이미 반영됨, 유지할 것)

- **Stage3 프레임률**: `runtime.count_frames`(ffprobe 실측) + `resample_indices(20→10Hz)`.
  컨테이너 fps 신뢰 금지 (docs/05). 이미 predict_stage3 에 반영.
- **Stage2 프레임 재번호 금지**: `data.frame_number` 가 파일명 숫자를 원본 번호로 사용.
- **제출 규격**: 반환 직전 `submit_guard.check_stage{1,2,3}` 통과 필수 (범주/정수/범위/STOPPED steer).
- **오프라인**: 모든 모델 `weights=None` 생성 후 `model/` 로컬 로드. hub 조회 금지.

## 7. 로컬(macOS)에서 검증 가능한 것 / 불가한 것

| 항목 | 로컬 macOS | 비고 |
|---|---|---|
| 구문/컴파일 검사 | ✅ | `python -m py_compile ...` |
| config dry-run (계약 검증) | ✅ | `METRIC dry_run_ok=1` |
| 20→10Hz 리샘플 로직 | ✅ | pandas 불필요 |
| validate_submission (repo) | ✅ | `METRIC submission_valid=1` |
| predict_*/submit_guard (pandas) | ❌ | 이 샌드박스 pandas 코드사인 정책으로 로드 불가 → GPU/Docker 기기에서 |
| 실제 학습/CUDA 추론 | ❌ | GPU 기기 필수 |

> 이 환경의 pandas 로드 제약은 알려진 한계다(코드 문제 아님). GPU/Docker 기기에서는
> `env/configs/requirements-train.txt` 스택으로 전부 정상 동작한다.

## 8. 검증 로그 (이 뼈대 스모크 테스트 결과)

```
py_compile           → OK (전 파일 구문 정상)
train --dry-run x3   → METRIC dry_run_ok=1 (stage1/2/3, 실제 config 값 반영)
resample_indices     → 20Hz 1200f → 10Hz 600샘플 (step=2) ✅
validate_submission  → METRIC submission_valid=1
```

---

## 실측 결과 (2026-08-29, exp/stage3-egomotion)

상세 로그: [`experiments/RESULTS.md`](experiments/RESULTS.md). 요약(majority baseline 대비):

| Stage | 지표 | 결과 | baseline | 개선 |
|---|---|---|---|---|
| 3 가감속 | accel_acc | 0.6417 | 0.5100 | +13.2pt |
| 3 조향 | steer_acc_moving | 0.8533 | 0.8433 | +1.0pt (약함) |
| 2 충돌 | collision_mae_sec | 0.2200 | 0.7800 | -5.6frame |
| 1 누설 | codec_leak | before 1 → after 0 | — | 중화 성공 |

재현: `python -m experiments.stage3_flow_eval` / `stage2_collision_eval` / `stage1_codec_leak_check`
