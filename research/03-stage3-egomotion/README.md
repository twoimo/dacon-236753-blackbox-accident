# Stage 3 — 차량 거동 특성 분석 (가감속·조향)

- **과업**: 실차 주행영상만 입력으로 **0.1초 단위** 가감속·조향 범주 산출.
  - `accel_label`: ACCELERATING / DECELERATING / CONSTANT / STOPPED
  - `steer_label`: LEFT / STRAIGHT / RIGHT
- **지표**: 범주 정확도. 조향은 **주행 중 유효 프레임만** 채점(STOPPED 제외) 하지만
  제출에는 전체 프레임 `steer_label` 필수. **가중치 0.4.**
- **근거 정본**: `research/references/README.md`.

---

## 1. 문제의 본질 — 영상만으로 ego-motion 추정

평가 시 CAN 데이터는 **입력으로 주어지지 않는다**(정답 생성에만 사용). 따라서 모델은
**영상으로부터 자차 운동(속도 변화·회전)** 을 추정해 범주로 분류해야 한다. 이는
self-supervised **depth + ego-motion** 계열이 검증된 골격이다.

- `zhou2017sfmlearner` (CVPR 2017 Oral, **≈2,321 IEEE / ≈2,869 S2**): 영상만으로 DepthNet +
  **PoseNet** 을 view-synthesis 자기지도로 학습. ego-motion(프레임 간 6-DoF pose) 추정의 원조.
- `godard2019monodepth2` (ICCV 2019, ≈4,000+ GS): 멀티스케일·오클루전 마스킹·최소 재투영 손실.
  ResNet-18 PoseNet 인코더가 우리 백본과 정합적.
- `bian2019scsfm` (NeurIPS 2019, ≈700 GS): **스케일 일관성** 손실 → 프레임 간 속도 스케일 드리프트
  억제. 가감속처럼 속도의 시간 미분이 중요한 과업에 유리.

**중요한 구분**: 위 논문들은 pose **회귀**다. 대회는 **범주 분류**(4-class / 3-class)이므로,
PoseNet/특징 추출기를 **사전학습 백본**으로 쓰고 그 위에 **분류 헤드**를 얹는 것이 실용적이다.
또는 optical flow(예: 프레임 간 흐름 크기→가감속, 흐름의 좌우 divergence→조향) 특징을 병용.

## 2. 데이터·라벨 근거 — comma2k19

- `schafer2018comma2k19` (arXiv 2018, MIT): 33시간·2,019 세그먼트, 카메라+9축 IMU+CAN+raw GNSS.
  우리 `data/external/comma2k19/` 의 CAN 속도·조향각이 Stage 3 라벨 파생의 정답 출처.
- 라벨 생성: `scripts/make_stage3_labels_from_comma2k19.py` 가 CAN을 프레임 시각으로 보간 →
  10Hz 리샘플 → 임계값으로 범주화.

### 조향 부호 규약 (실측 검증 필수)
- openpilot 규약: **양수 조향각 = 좌회전**. 매 실행 검증(ECEF 속도의 ENU 헤딩 변화율과 상관).
  예제 세그먼트에서 상관 **−0.374** (ENU atan2(E,N) 양의 방향=우회전 → 음의 상관 = 양수 조향=좌회전).
  부호가 뒤집히면 스크립트가 라벨을 만들지 않고 종료(안전장치).

### 임계값은 추정치 — 반드시 튜닝
- `docs/04-datasets.md` 표: STOPPED(<0.5 m/s), ACCEL/DECEL(±0.3 m/s²), LEFT/RIGHT(±1.0°).
- 대회 정답 생성 규칙이 비공개라 **추정치**다. 고속도로에서 `|조향각|` 중위수 0.4°, 90퍼센타일 1.2°로
  매우 작아, 조향 임계값 1.0°가 분포를 크게 바꾼다. → **임계값 스윕이 핵심 실험.**

## 3. 프레임률 함정 (docs/05 반영, 치명적)

- Stage 3 영상은 실제 **20Hz**인데 컨테이너가 **40fps로 오선언**, 후반 PTS 손상으로 `duration` 신뢰 불가.
- 대회 스펙 라벨은 **10Hz**(0.1초 단위).
- 프레임 수는 반드시 `ffprobe -count_frames` 로 **실측**. 프레임마다 `sample_index`를 발급하면
  시간축이 2배 어긋난다. → 20Hz 실측 → 10Hz 리샘플 매핑을 명시적으로 관리.

## 4. 권장 접근 (근거 기반)

### 백본 / 특징
- **경로 A**: Monodepth2 스타일 PoseNet(ResNet-18) 특징 + 시간 윈도우 → 가감속/조향 분류 헤드.
- **경로 B**: 프레임 간 optical flow(RAFT/경량) 통계 특징 → 소형 시계열 분류기. 해석 쉬움·경량.
- **경로 C**: X3D-S 클립 분류(멀티태스크: accel 4-class + steer 3-class). 베이스라인과 정합.
- 세 경로 앙상블 여지. 60분 제약상 경량(B/C) 우선.

### 학습 데이터 확보
- 예제 세그먼트 1개는 600행만 나온다. **전체 comma2k19(~100GB)** 받으면 2,019×600 ≈ 120만 행.
  이것이 Stage 3 현실적 학습 경로.
- 라이선스: comma2k19 MIT (대회 "법적 제한 없는 데이터" 요건 충족). 출처 기록 유지.

## 5. 검증 가능한 실험 후크
- `env/configs/stage3.yaml` 의 `label_thresholds`(accel/steer 임계값 스윕), `resample_hz`,
  `frame_count_method: ffprobe_count_frames`.
- 하니스 지표: `METRIC accel_acc=<value>` + `METRIC steer_acc_moving=<value>`
  (STOPPED 제외 조향 정확도). 채점 규약과 일치.

## 6. 우선순위
1. 프레임률(20→10Hz) 매핑 정확화 — 여기서 틀리면 나머지 무의미.
2. 조향 부호·임계값 검증 및 스윕.
3. 백본 경로 B/C 비교 후 앙상블.
