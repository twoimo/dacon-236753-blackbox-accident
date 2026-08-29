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
