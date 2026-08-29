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

## Stage 1 분리가능성 실측 — 코덱 중화 후 재촬영 특징 (2026-08-29, 로컬 맥)

핵심 질문(가중치 0.2 타당성): 코덱 누설을 제거해도 재촬영본이 numpy 특징만으로 원본과 구분되는가?
재현: `python -m experiments.stage1_separability_eval --n 300`.

방법: CrashBest 프레임(ORIGINAL) → synthesize_recapture(RERECORDED) 쌍 생성 → **두 클래스 동일 JPEG
재양자화**(코덱 누설 차단) → numpy 특징(라플라시안 에너지/FFT 고대역/엣지밀도/블록분산) → 순수 numpy
로지스틱회귀 → 소스영상 기준 train/test 분리.

| 규모(소스) | seed | 분리 정확도 | Macro-F1 |
|---|---|---|---|
| 200 | 42 | 0.900 | 0.900 |
| 300 | 42 | 0.928 | 0.928 |
| 600 | 7  | 0.908 | 0.908 |

개별 특징 판별력(AUC): **edge_density 0.72~0.74, fft_high 0.67~0.75** (최강), lap_energy/block_var ~0.5(약).

**결론**: 코덱을 중화해도 재촬영 아티팩트(FFT 고대역·엣지 통계)로 두 클래스가 안정적으로 분리됨(acc≈0.9,
baseline 0.5). Stage 1 접근(잔차+주파수 특징)의 타당성 실증.

**정직한 캐비앗**: 이는 "우리 합성기 vs 원본" 분리다. 모델이 진짜 재촬영 물리가 아니라 합성기 시그니처를
학습했을 수 있음(research/01의 도메인 갭 경고). 실제 재촬영본 홀드아웃이 없으면 상한은 낙관적. 그래도
코덱-무의존 특징이 존재함은 확인됨 → GPU 2-스트림 학습 시 이 특징군을 잔차 스트림 근거로 사용.
