# 실험 결과 로그 (exp/stage3-egomotion)

autoresearch 전략을 코드로 구현하고 로컬에서 실측한 결과. 모든 수치는 재현 가능하며
majority-class baseline과 비교한다. 실행 환경: 로컬 macOS `.venv`(numpy+PIL+ffmpeg;
cv2/torch/pandas는 코드사인 정책으로 미로드 → GPU/Docker에서 최종 확인).

## 요약 (2026-08-29)

| Stage | 지표 | 결과 | Baseline | 개선 | 판정 |
|---|---|---|---|---|---|
| 3 accel | accel_acc | **0.6417** | 0.5100 (CONSTANT) | **+13.2pt** | keep |
| 3 steer | steer_acc_moving | 0.8533 | 0.8433 (STRAIGHT) | +1.0pt | weak |
| 3 종합 | mean_acc | 0.7475 | — | — | keep |
| 2 충돌 | collision_mae_sec | **0.2200** (2.2f) | 0.7800 (midpoint) | **-5.6f** | keep |
| 1 누설 | codec_leak | before=1 → **after=0** | — | 중화 성공 | keep |

## Stage 3 — 옵티컬플로우 ego-motion (무학습)

- 방법: 프레임 간 Lucas-Kanade 전역 흐름. 수직 흐름 크기 → 속도 프록시 → SG류 미분 → 가감속;
  수평 흐름(지평선 밴드 열별 중앙값) → 요레이트 → 조향.
- 실측: `accel_acc=0.6417`(+13.2pt), `steer_acc_moving=0.8533`(+1.0pt), `mean_acc=0.7475`.
- 검증: comma2k19 CAN 정답을 물리 기준으로 사용. CAN accel↔라벨 corr 0.79(상한), 영상 속도프록시↔CAN 0.30~0.34.
- 근거: Longuet-Higgins–Prazdny 1980(병진/회전 분해, 회전은 깊이 무관), Farnebäck 2003, Savitzky-Golay 1964.
- 실행: `.venv/bin/python -m experiments.stage3_flow_eval`
- 핵심 함정 수정: OPEN_*.mp4는 40fps 오선언+PTS 손상 → `-fps_mode passthrough` 후 stride-2로 10Hz 정합(docs/05).
- 캐비앗: 조향은 majority 겨우 상회(고속도로 세그먼트라 STRAIGHT 84%). STOPPED 0개(예측 안 함). 단일 세그먼트 튜닝.

## Stage 2 — 충돌 시간 국소화

- 방법: 프레임간 모션 피크 argmax + **CCD 시간 prior**(충돌은 항상 클립 후반 ≥0.60N).
- 실측: `collision_mae=2.2 frames (0.22s)` vs midpoint 7.8f(0.78s), 5.6프레임 개선.
- 핵심 발견: 1,500 CCD 영상 전수 확인 결과 first_crash_frame/N ≥ 0.60(중위 0.72) → 후반 탐색 prior가
  카메라 워밍업 스파이크 오탐(000002: 4→35f 교정)을 제거.
- 실행: `.venv/bin/python -m experiments.stage2_collision_eval`
- 캐비앗: 로컬 라벨 5개뿐(추정기 검증용). entry/evasion/side는 정답 부재로 미검증(약지도).

## Stage 1 — 코덱 누설 차단 + 재촬영 합성

- 실측: `codec_leak_before=1`(ORIGINAL=mpeg4/RERECORDED=h264로 코덱이 라벨 완전예측) →
  두 클래스 동일 재인코딩(libx264 crf23 gop30) 후 `codec_leak_after=0`(코덱 동일, 누설 제거).
- 재촬영 합성기: 서브픽셀 격자+감마+미세 호모그래피+MTF 블러+리프레시 밴딩+센서노이즈(클립마다 랜덤).
- 근거: Wang&Farid 2006, Jiang TIFS 2019(동일 파라미터도 흔적 남음 경고), research/01.
- 실행: `.venv/bin/python -m experiments.stage1_codec_leak_check` / `stage1_synth_recapture.py`
- 캐비앗: 실제 2-스트림 학습은 GPU 필요(torch 게이팅). 현재는 누설 진단 + 합성 파이프라인까지.

## 다음 단계

1. Stage 3: 전체 comma2k19(~2000 세그먼트)로 라벨 확장 → 조향 신호 강화(현재 단일 세그먼트 한계).
2. Stage 2: CCD first_crash_frame 지도로 ResNet-18→BiGRU 충돌 헤드 GPU 학습.
3. Stage 1: CrashBest에서 재촬영 합성 대량 생성 → LODO 검증으로 2-스트림 학습.
4. GPU 기기에서 pandas/cv2/torch 경로 최종 확인 → submit.zip 패키징.

---

## Stage 2 대규모 실측 — CCD 1,500영상 전체 (2026-08-29, 로컬 맥)

로컬 맥(numpy+PIL, torch/cv2 불가)에서 가능한 최대 규모 검증. 기존 5샘플 → **1,500영상 전체**
(catalog first_crash_frame_index 정답). 재현: `python -m experiments.stage2_ccd_large_eval`.

| 방법 | MAE(frames) | MAE(sec@10fps) | median | within-3f |
|---|---|---|---|---|
| midpoint (baseline) | 12.19 | 1.219 | 11.0 | — |
| argmax (전역 피크) | 11.11 | 1.111 | 7.0 | — |
| windowed (prior 0.55) | 5.58 | 0.558 | 4.0 | 0.45 |
| **onset + prior 0.60 (채택)** | **5.22** | **0.522** | 4.0 | **0.48** |

튜닝(experiments/stage2_estimator_search.py, prior 스윕):
- prior_frac 스윕 0.45~0.70 → **0.60 최적** (5.41f vs 0.55의 5.53f).
- 추정 방식: **onset(모션 1차차분 최대=충돌 상승엣지)** 이 window(피크 argmax)보다 우수, ego:No(제3자 사고)에서 특히.

환경별(onset, window 유사): Snowy 4.37f < Day 5.62f < Rainy 6.10f. ego:Yes(자차관여) 4.37f < ego:No 6.97f.

**결론**: 무학습 모션 휴리스틱의 통계적 상한은 MAE ~5.2f(0.52s), within-3f 48%. midpoint 대비 절반 이하로 개선.
5샘플(2.2f)은 낙관적 소표본이었음(정직). 추가 개선은 research/synthesis대로 **CCD 지도 학습(ResNet-18→BiGRU, GPU 필요)** 이 필요.
채택 config를 src/stage2/predict.py에 반영(CCD_MIN_FRAC=0.60, USE_ONSET=True).

---

## Stage 1 분리가능성 실측 — 코덱 중화 후 numpy 특징 (2026-08-29, 로컬 맥)

핵심 질문: 재촬영 예제가 합성본이고 코덱 누설을 제거하면, **픽셀 특징만으로 원본 vs 재촬영이
실제로 구분되는가?** (가중치 0.2 컴포넌트 타당성). 재현: `python -m experiments.stage1_separability_eval --n 300`.

절차: CrashBest 원본 + synthesize_recapture 합성본 쌍 → 두 클래스 **동일 JPEG 재양자화**(코덱 누설 차단)
→ numpy 특징(고주파 잔차, FFT 고대역, 엣지밀도, 블록분산) → 순수 numpy 로지스틱회귀 → 소스분리 홀드아웃.

| 규모 | seed | acc | Macro-F1 |
|---|---|---|---|
| 300소스(600표본) | 42 | 0.928 | 0.928 |
| 600소스(1200표본) | 7 | 0.908 | 0.908 |

개별 특징 판별력(AUC): **edge_density 0.73~0.74, fft_high 0.70~0.75** (재촬영의 MTF 블러+모아레 흔적),
lap_energy/block_var ~0.5(약함).

**결론**: 코덱 누설을 제거해도 재촬영 아티팩트(엣지 번짐·FFT 고대역)로 **~91% 분리 가능** →
Stage 1은 픽셀 특징 기반으로 충분히 학습 가능(research/01 가설 실증). baseline(항상 한 클래스) Macro-F1 0.33 대비 큰 폭 우위.
캐비앗: 재촬영이 우리 합성기 산출물이라 실제 재촬영 도메인과 갭 존재(research/01 §도메인시프트) —
실제 성능은 물리 재촬영 데이터 + leave-one-device-out 검증 필요. 하지만 "분리 신호가 존재한다"는 핵심은 확인됨.

---

## Stage 2 학습형 충돌 헤드 — GPU(MPS) 학습 (2026-08-30)

보안 정책 해제로 **M5 Max 40코어 GPU(MPS) 활성화** (torch 2.8.0, CPU 대비 matmul 5.3배).
CCD 1,500영상 특징(32×32 flatten + 모션)으로 학습형 충돌 국소화 헤드를 학습.

### 모델 진화
| 버전 | 구조 | test MAE(frame) | 비고 |
|---|---|---|---|
| v1 | Linear+1D conv, motion 1채널 | 6.31 | 휴리스틱(5.22)에 **패배** |
| v2 | BiGRU + motion/onset 2채널 | 5.25 | 휴리스틱 근접 |
| **v2 앙상블(50/50)** | 학습 + 모션 onset | **4.54** | 휴리스틱 **초과** |

### 5-fold 교차검증 (video split, 최종)
| 방법 | MAE(frame) | MAE(sec) |
|---|---|---|
| heuristic onset | 5.68 ± 0.28 | 0.568 |
| learned head | 4.74 ± 0.29 | 0.474 |
| **ensemble (채택)** | **4.54 ± 0.12** | **0.454** (within-3f 0.50) |

- 휴리스틱 대비 **1.14프레임(0.11초) 개선**, 분산 작아(±0.12) 안정적.
- 재현: `python -m experiments.stage2_cv_train --epochs 60` (GPU ~100초).
- 학습 특징: `python -m experiments.stage2_extract_features` (75MB npz 캐시).
- 모델 저장: `model/stage2/collision_head.pt` (MotionAwareHead, BiGRU emb96/gru96).
- **predict 통합**: `src/stage2/predict.py` 가 collision_head.pt 로드 → 학습 logit(prior창 argmax)과
  모션 onset 을 50/50 앙상블. 로컬 5샘플·CCD 20영상 실행 검증 완료(제출 규격 통과).

**정직한 캐비앗**: MPS 미가용 환경(평가서버 CUDA)에서는 predict 의 numpy/heuristic 경로로도 동작하도록
설계됨(torch 있으면 학습헤드, 없으면 휴리스틱 폴백). 학습 특징이 raw 32×32 flatten이라 CNN 공간특징
대비 단순 → 여기서 더 낮추려면 프레임 CNN 백본(ResNet-18) 학습이 다음 단계(GPU 시간 더 필요).

---

## Stage 2 학습형 충돌 헤드 — GPU(MPS) 학습 (2026-08-30, 로컬 맥 M5 Max)

보안 정책 해제로 **MPS GPU 활성화**(CPU 대비 5.3배). torch 2.8.0/pandas/cv2 로드 가능.
CCD 1,500영상 특징(32×32 grayscale flatten 1024 + 모션)으로 충돌 국소화 헤드를 GPU 학습.
재현: `python -m experiments.stage2_extract_features` → `python -m experiments.stage2_cv_train`.

시행착오(정직 기록):
- v1 (temporal conv, 특징만): MAE 6.31f → **휴리스틱(5.22f)보다 나쁨** (모션 신호 미활용).
- v2 (BiGRU + 모션/onset 명시 채널): 개선. 학습+휴리스틱 앙상블이 최고.

**5-fold 교차검증 (video split, 1,500영상):**

| 방법 | MAE(frame) | MAE(sec) | within-3f |
|---|---|---|---|
| heuristic onset+prior | 5.68 ± 0.28 | 0.568 | — |
| learned head (BiGRU) | 4.74 ± 0.29 | 0.474 | — |
| **ensemble 50/50 (채택)** | **4.54 ± 0.12** | **0.454** | **0.50** |

- 앙상블이 휴리스틱 대비 **1.14프레임(0.11초) 개선**, 분산 작아(±0.12) 안정적.
- 최종 모델: `model/stage2/collision_head.pt` (전체 데이터 학습). predict_stage2 가 로드해
  학습 logit(prior 창 argmax) + 모션 onset 을 50/50 앙상블. 없으면 heuristic 폴백.
- predict 파이프라인 실행 검증 완료(5 샘플 + CCD 20영상 MAE 2.3f/학습셋 참고치).

**GPU 진단**: 이전 세션의 MPS 차단은 샌드박스 sysctl 차단이 원인이었고, 보안 해제 후
`mps.is_available()=True`. M5 Max 40코어 GPU로 학습(60ep×5fold ≈ 100초). 평가서버는 CUDA L40S
이므로 가중치 저장 후 서버 추론 재현 필요(weights 로컬 로드 규약).

**정직한 캐비앗**: 32×32 flatten 은 공간구조 손실 — CNN(ResNet-18) 특징이면 더 오를 여지.
CCD 라벨은 first_crash_frame_index 로 정의됐고 대회 비공개 정답 규칙과 다를 수 있음(추정).
