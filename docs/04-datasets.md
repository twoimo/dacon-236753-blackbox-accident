# 데이터셋

대회는 **학습 데이터셋을 제공하지 않는다.** 배포되는 것은 입출력 형식 확인용 공개 예제
15건뿐이며, 참가자가 법적 제한이 없는 학습 데이터를 직접 구성해야 한다. 2차 평가에서
'학습데이터 구성 보고서'를 요구하는 이유이기도 하다.

기계 판독용 정본은 [`catalog/catalog.json`](../catalog/catalog.json) 이다. 아래 수치는
모두 그 카탈로그에서 나온 실측값이며, 사람이 읽기 위한 요약이다.

---

## 1. 로컬 레이아웃

평가 서버가 `data_dir` 아래에 구성하는 구조를 **그대로 미러링**한다. 따라서
`baseline/baseline_train.ipynb` 의 `DATA = ROOT/'data'` 가 수정 없이 동작한다.

```text
data/
├── stage1/                       # 재녹화 판별
│   ├── labels.csv                #   ID, path, label
│   └── videos/
│       ├── original/    000001..000005.mp4     (mpeg4, 1280x720, 10fps, 50f)
│       └── rerecorded/  000001..000005.mp4     (h264,  1280x720, 10fps, 50f)
├── stage2/                       # 사고 주요시점·상황
│   ├── labels.csv                #   ID, path, t_collision, t_entry, evasion_space, entry_side
│   ├── videos/          000001..000005.mp4     (mpeg4, 1280x720, 10fps, 50f)
│   └── images/          <ID>/frame_000000.jpg .. frame_000049.jpg
│                                 #   평가 서버와 동일한 입력 형태. `make stage2-images` 로 생성.
├── stage3/                       # 차량 거동
│   ├── labels.csv                #   ※ 로컬에 없음 — 05-data-integrity-report.md 참고
│   ├── labels_comma2k19.csv      #   CAN 정답에서 파생한 대체 라벨 (600행)
│   └── videos/          OPEN_001..OPEN_005.mp4 (hevc, 1164x874, ~1200f)
└── external/                     # 외부 학습 데이터셋
    ├── CrashBest/                #   CCD 크래시 프레임 75,000장
    ├── Crash_Table.csv           #   CCD 메타데이터 1,500행
    └── comma2k19/                #   comma2k19 저장소 + Example 세그먼트
```

`data/external/`, `data/stage3/videos/`, `data/stage2/images/` 는 깃에 없다.
[07-dataset-distribution.md](07-dataset-distribution.md) 의 링크에서 `make data` 로 복원한다.

---

## 2. 대회 공개 예제

출처: 대회 배포 `baseline.zip`. 15개 영상 + 라벨.

| Stage | 구성 | 정답 범위 |
|---|---|---|
| 1 | 원본 5 + 재녹화 5 | 전체 |
| 2 | 사고 영상 5 | **`t_collision` 만** — `t_entry`·`evasion_space`·`entry_side` 는 `-1` |
| 3 | 주행 영상 5 | 희소 라벨 (로컬 소실) |

원본 유래 (`data/SOURCES.md` 원문 기준):

- Stage 1·2 원본: CCD `Crash-1500.zip` 의 `000001`~`000005`
- Stage 1 재녹화: 위 원본 5건에 화면 재촬영 시 나타나는 리샘플링·노이즈 특성을 적용한 **파생본**.
  실제 다른 기기로 재촬영한 데이터가 아니다.
- Stage 2 충돌 구간: CCD 공식 `Crash-1500.txt` 주석의 첫 positive frame
- Stage 3: comma2k19 공개 데이터에서 생성한 5개 주행 클립 + CAN 기반 라벨

이 유래 때문에 **Stage 1 의 ORIGINAL 파일과 Stage 2 의 영상은 바이트 단위로 동일**하다.
카탈로그는 이를 "의도된 중복"으로 분류한다.

> 재녹화 예제가 진짜 재촬영본이 아니라는 점은 Stage 1 설계에 직접 영향을 준다.
> 공개 예제만으로 학습하면 코덱 차이(mpeg4 vs h264) 같은 인공적 단서를 배우게 되고,
> 실제 평가 데이터에서는 무너진다. 자세한 내용은 [05](05-data-integrity-report.md) 참고.

---

## 3. CCD (Car Crash Dataset)

| 항목 | 값 |
|---|---|
| 경로 | `data/external/CrashBest/`, `data/external/Crash_Table.csv` |
| 이미지 | **75,000장** = 1,500 영상 × 50 프레임 (결측 0) |
| 용량 | 8.0 GiB |
| 해상도 | 1280×720 **73,250장** / 960×720 800장 / 640×360 950장 |
| 메타데이터 조인 | 1,500/1,500 (100%) |
| 상위 저장소 | https://github.com/Cogito2012/CarCrashDataset |
| Kaggle 미러 | https://www.kaggle.com/datasets/asefjamilajwad/car-crash-dataset-ccd |
| 논문 | Bao et al., *Uncertainty-based Traffic Accident Anticipation*, ACM MM 2020 |

해상도는 **영상 단위로는 일관**하고(한 영상 안에서 섞이지 않음), 1,500 영상 중 35개만
비주류 해상도다(960×720 16개, 640×360 19개).

### 파일명 규약

```
C_<video_id>_<frame_no>.jpg      예: C_000001_33.jpg
                                  video_id = 000001 (Crash_Table.csv 의 vidname)
                                  frame_no = 33     (1-기반, 1..50)
```

### `Crash_Table.csv` 컬럼

| 컬럼 | 의미 |
|---|---|
| `vidname` | 영상 ID (`000001`~`001500`) |
| `frame_1` … `frame_50` | 프레임별 크래시 여부. `0`=비충돌, `1`=충돌 |
| `startframe` | 원본 YouTube 영상에서의 시작 프레임 오프셋 |
| `youtubeID` | 출처 YouTube 식별자 |
| `timing` | `Day` 1,325 / `Night` 175 |
| `weather` | `Normal` 1,141 / `Snowy` 235 / `Rainy` 124 |
| `egoinvolve` | 자차 관여 여부. `Yes` 801 / `No` 699 |

**첫 positive frame** 이 Stage 2 의 `t_collision` 정의다.
`catalog/crashbest_videos.csv` 의 `first_crash_frame_index` 컬럼에 1,500개 전부
0-기반으로 미리 계산돼 있다.

### 활용

- **Stage 1**: 원본 dashcam 프레임 공급원. 재녹화 클래스는 화면 재촬영을 모사한
  증강(모아레, 리샘플링, 재압축, 밝기·감마 변화, 미세 기하 변형)으로 직접 만들어야 한다.
- **Stage 2**: 프레임별 0/1 주석이 충돌 시점 시계열 국소화(temporal localization) 학습에 그대로 쓰인다.
  `entry_frame`·`evasion_space`·`entry_side` 정답은 없으므로 별도 라벨링이 필요하다.

### 아직 확보하지 않은 CCD 구성요소

상위 저장소의 전체 배포본에는 로컬에 없는 항목이 더 있다.

| 항목 | 내용 | 용도 |
|---|---|---|
| `videos/Crash-1500/` | 1,500개 크래시 MP4 원본 | 프레임이 아닌 영상 단위 학습 |
| `videos/Normal/` | 3,000개 정상 주행 MP4 (BDD100K 표본) | Stage 2 negative 표본 |
| `vgg16_features/` | 사전 추출 VGG-16 특징 (.npz) | 빠른 실험 |

---

## 4. comma2k19

| 항목 | 값 |
|---|---|
| 경로 | `data/external/comma2k19/` |
| 로컬 용량 | 64 MiB (`.git` 제외) |
| 상위 저장소 | https://github.com/commaai/comma2k19 |
| 전체 데이터셋 | Academic Torrents, ~100GB, 2,019 세그먼트, 33시간+ |
| 구간 | CA-280 고속도로 (San Jose ↔ San Francisco) |

로컬에는 예제 세그먼트 1개가 있다:
`Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/`

| 경로 | 형태 | 검증 결과 |
|---|---|---|
| `video.hevc` | HEVC 1164×874 | 1,200 프레임 = **20Hz × 60초** |
| `global_pose/frame_times` | (1200,) | 프레임별 시각. `dt` 평균 0.0500s |
| `global_pose/frame_positions` | (1200, 3) | ECEF 위치 |
| `global_pose/frame_velocities` | (1200, 3) | ECEF 속도 |
| `global_pose/frame_orientations` | (1200, 4) | 쿼터니언 |
| `processed_log/CAN/speed/{t,value}` | (4974,), (4974,1) | ~83Hz. 7.97 – 19.84 m/s |
| `processed_log/CAN/steering_angle/{t,value}` | (4974,) | −4.60 – +2.37 deg |
| `processed_log/CAN/wheel_speed/{t,value}` | (4974,), (4974,4) | 바퀴별 속도 |
| `processed_log/IMU/accelerometer/{t,value}` | (6256,), (6256,3) | ~104Hz |
| `processed_log/GNSS/…` | 가변 | qcom/ublox, raw/live |

> `openpilot/` 서브모듈이 비어 있다(0바이트). `utils/` 의 일부 헬퍼가 이를 필요로 한다.
> 필요하면 `git -C data/external/comma2k19 submodule update --init` 로 채운다.

### 조향 부호 규약 (실측 검증)

openpilot 규약대로 **`steering_angle` 양수 = 좌회전**이다. 가정이 아니라 매 실행마다
검증한다. `global_pose` 의 ECEF 속도를 지역 ENU 평면에 투영해 헤딩 변화율을 구한 뒤
`steering_angle` 과의 상관계수를 계산하는데, 예제 세그먼트에서 **−0.374** 가 나온다.
ENU `atan2(E, N)` 기준 양의 방향은 시계방향(우회전)이므로, 음의 상관은 곧
"양수 조향각 = 좌회전"을 뜻한다. 부호가 뒤집히면
`scripts/make_stage3_labels_from_comma2k19.py` 는 라벨을 만들지 않고 종료한다.

### 활용 — Stage 3 라벨 생성

```bash
.venv/bin/python scripts/make_stage3_labels_from_comma2k19.py
```

CAN 속도·조향각을 영상 프레임 시각으로 보간한 뒤 대회 스펙인 10Hz 로 리샘플링해
`data/stage3/labels_comma2k19.csv` 를 만든다. 예제 세그먼트 기준 600행이 나오고 분포는
`CONSTANT` 306 / `ACCELERATING` 155 / `DECELERATING` 139,
`STRAIGHT` 506 / `RIGHT` 53 / `LEFT` 41 이다.

판정 규칙(전부 CLI 옵션으로 조정 가능):

| 라벨 | 규칙 | 기본값 |
|---|---|---|
| `STOPPED` | `speed < --stop-speed` | 0.5 m/s |
| `ACCELERATING` | `dv/dt ≥ +--accel-threshold` | 0.3 m/s² |
| `DECELERATING` | `dv/dt ≤ −--accel-threshold` | 0.3 m/s² |
| `CONSTANT` | 그 외 | |
| `LEFT` | `angle ≥ +--steer-threshold` | 1.0 deg |
| `RIGHT` | `angle ≤ −--steer-threshold` | 1.0 deg |
| `STRAIGHT` | 그 외 | |

임계값은 대회 정답 생성 규칙이 공개되지 않았으므로 **추정치**다. 고속도로 주행에서
`|조향각|` 의 중위수는 0.4°, 90퍼센타일은 1.2° 로 매우 작기 때문에 조향 임계값 1.0° 는
결과 분포를 크게 바꾼다. 실제 학습에서는 반드시 이 값을 튜닝해야 한다.

전체 데이터셋(~100GB)을 받으면 같은 스크립트가 2,019 세그먼트 × 600행 =
약 120만 행의 라벨을 생성하며, 이것이 Stage 3 의 현실적인 학습 데이터 확보 경로다.

---

## 5. 라이선스와 사용 조건

| 데이터 | 조건 |
|---|---|
| 대회 공개 예제 | 대회 참가 목적 |
| CCD | 상위 저장소 조건 확인 필요. YouTube 출처 영상 기반 |
| comma2k19 | MIT (저장소 `LICENSE` 참고) |

대회 규정상 학습 데이터는 **사용에 법적 제한이 없어야** 하고, 2차 평가에서 구성 내역을
보고해야 한다. 새 데이터를 추가할 때마다 출처·라이선스를 이 문서에 기록해 두는 편이 좋다.
