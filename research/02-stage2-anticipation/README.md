# Stage 2 — 사고 주요시점·상황 분석

- **과업**: 사고 영상(프레임 이미지 폴더 입력)에서 4가지 산출.
  - `collision_frame`: 충돌 프레임
  - `entry_frame`: 피해차량 최초 진입 프레임
  - `evasion_space`: 회피 공간 여부 (0/1)
  - `entry_side`: 진입 방향 (LEFT/RIGHT)
- **지표**: 프레임→초 변환 후 정답 시각과 비교(시각 오차) + 범주 정확도. **가중치 0.4.**
- **근거 정본**: `research/references/README.md`.

---

## 1. 이 과업은 "예측(anticipation)"이 아니라 "국소화(localization)"에 가깝다

학계 dashcam 문헌 대부분은 사고가 나기 **전에** 예측하는 anticipation 과업이다
(`chan2016anticipating` ACCV 2016 ≈410 cites, `bao2020ccd` ACM MM 2020). 반면 대회 Stage 2는
**이미 일어난 사고 영상**에서 충돌/진입 시점을 프레임 단위로 찾는 **시간 국소화**다.

따라서:
- anticipation 문헌은 **특징 설계·주석 정의·데이터** 참고용(특히 CCD가 여기서 왔다).
- 실제 헤드는 프레임 단위 **시계열 회귀/분류**(peak detection / temporal localization)로 설계한다.
  최근 벤치마크도 충돌시점을 "peak prediction"(가우시안 유사도로 시각 오차를 부드럽게 벌점)으로
  다룬다 — 대회 채점(초 단위 오차)과 정합적이다.

## 2. 데이터 근거 — CCD/CrashBest

- `bao2020ccd`: Car Crash Dataset. **프레임별 0/1 충돌 주석**이 핵심. 우리 `Crash_Table.csv`의
  `frame_1..frame_50` 이 그것이고, **첫 positive frame** 이 `t_collision` 정의다
  (`catalog/crashbest_videos.csv` 의 `first_crash_frame_index` 에 1,500개 계산됨).
- **주의**: CCD는 `entry_frame` / `evasion_space` / `entry_side` 정답을 제공하지 않는다.
  이 세 항목은 **별도 라벨링 전략**이 필요하다 (2차 평가 '학습데이터 구성 보고서' 대상).

## 3. 항목별 권장 접근

### 3.1 collision_frame — 시간 국소화 (근거 가장 탄탄)
- 프레임별 "충돌 확률" 시계열 → argmax 또는 change-point.
- 백본: 경량 2D CNN(프레임 특징) + 시간 모듈(BiGRU/temporal conv) 또는 X3D 클립.
  베이스라인이 ResNet-18→BiGRU 를 쓰므로 호환 경로 존재.
- 손실: 충돌 시각 주변을 부드럽게 벌하는 가우시안/soft-label 회귀가 대회 채점(초 오차)과 정합.

### 3.2 entry_frame — 진입 시점
- 피해차량이 자차 차선에 최초 진입하는 순간. 정답 라벨이 CCD에 없으므로,
  **객체 추적(tracklet) + 차선 기하** 로 약지도(weak-label)를 생성해야 한다.
- 근거: `zeng2017agentcentric` (CVPR 2017) — agent-centric 위험영역 국소화.
  피해차량 bbox 궤적이 ego 차선 영역에 들어오는 프레임을 진입으로 정의.

### 3.3 entry_side — 진입 방향 (LEFT/RIGHT)
- 피해차량 궤적의 화면 좌/우 기원으로 분류. bbox center-x 의 진입 직전 부호가 강한 신호.
- 단순 규칙(추적 기반) + 소형 분류 헤드 앙상블 권장. 정답 없음 → 약지도.

### 3.4 evasion_space — 회피 공간 여부 (0/1)
- 충돌 시점 자차 주변 여유 공간 유무. 자유공간/차선 세그멘테이션 또는 주변 차량 밀도로 추정.
- 가장 라벨이 어려운 항목. 초기엔 heuristic 베이스라인 → 이후 약지도 개선.

## 4. 채점 함정 (docs/03 반영)

다음은 전부 **오답 처리**된다. 후처리에서 반드시 방어:
- 예측 누락 / 결측·비수치 / 음수 프레임 / 영상 범위 초과 프레임 / 허용 외 범주값.
- 프레임 번호는 **파일명 숫자 그대로**(재번호 금지). 로컬은 0-기반(`frame_000000~`)이라
  `labels.csv` 의 `t_collision` 과 제출 `collision_frame` 이 같은 좌표계.

## 5. 검증 가능한 실험 후크
- `env/configs/stage2.yaml` 의 `collision_head` / `entry_labeling` / `postprocess_clip` 블록.
- 하니스 지표: `METRIC time_err_sec=<value>` (충돌/진입 초 오차) + `METRIC cat_acc=<value>`
  (evasion/side 범주 정확도). 값이 낮을수록(오차) / 높을수록(정확도) 개선.

## 6. 우선순위
1. **collision_frame** 완성 (근거 탄탄, 가장 확실한 점수).
2. entry_frame 약지도 파이프라인.
3. entry_side (규칙+분류).
4. evasion_space (가장 불확실 — 보고서용 실험 기록 중요).
