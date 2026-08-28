# 데이터 무결성 점검 보고

최초 점검: 2026-08-28 · 도구: `scripts/build_catalog.py`, `scripts/verify_integrity.py`

기계 판독용 결과는 [`catalog/integrity.json`](../catalog/integrity.json) 에 있다.
이 문서는 그 판정의 **근거와 조치 내용**을 기록한다.

```bash
make catalog   # 재점검 (스캔 + 판정, 약 1분)
make check     # 카탈로그 대조 + 요약 출력
```

---

## 요약

| 심각도 | 건수 | 항목 |
|---|---|---|
| error | 1 | Stage 3 라벨 소실 (복구 불가) |
| warning | 2 | Stage 3 프레임레이트 불일치, CrashBest 해상도 혼재 |
| info | 1 | CrashBest 연속 프레임 중복 (정상) |
| ok | 8 | 나머지 계약 충족 |

---

## 1. 수정 완료: Stage 1 ORIGINAL 클래스가 RERECORDED 복사본으로 덮여 있었음

### 증상

`videos/original/` 과 `videos/rerecorded/` 의 5개 파일이 **모두 바이트 단위로 동일**했다.

```
000001  orig md5=9952388e…  rere md5=9952388e…  동일
000002  orig md5=185eb08e…  rere md5=185eb08e…  동일
000003  orig md5=fd42d1bd…  rere md5=fd42d1bd…  동일
000004  orig md5=514c7949…  rere md5=514c7949…  동일
000005  orig md5=1c363d02…  rere md5=1c363d02…  동일
```

`labels.csv` 는 같은 입력에 `ORIGINAL` 과 `RERECORDED` 를 동시에 부여하고 있었다.
Stage 1 학습이 구조적으로 불가능한 상태였고, 조용히 실패하는 종류의 문제였다.
`fit_stage1()` 은 정상 종료하지만 학습되는 것이 없다.

### 복구 근거

소실된 것은 ORIGINAL 쪽이었다. 세 가지 증거가 일치했다.

1. 기존 `data/data_index.json` 이 기록한 ORIGINAL 파일 크기
   `526859, 286879, 427410, 305244, 363811` 이 당시 `samples/stage2/*.mp4` 의 크기와
   **5/5 정확히 일치**했다.
2. `data/SOURCES.md` 가 "Stage 1·2 원본은 CCD `Crash-1500.zip` 의 `000001`~`000005`" 라고
   명시한다. 즉 두 Stage 가 같은 원본 클립을 공유하는 것이 설계다.
3. 코덱이 갈린다. Stage 2 영상과 소실된 ORIGINAL 은 **mpeg4**, 남아 있던 두 디렉터리는
   모두 **h264** 였다. 재녹화 파생본 생성 시 h264 로 재인코딩한 흔적이다.
4. 프레임을 직접 눈으로 대조했다. `stage2/000001.mp4` 의 0번 프레임과
   `CrashBest/C_000001_01.jpg` 는 같은 장면(2010.02.18 07:02:02 타임스탬프 오버레이,
   노란 보닛)이었다.

### 조치

`data/stage2/videos/*.mp4` 를 `data/stage1/videos/original/` 로 복원했다.

```
000001  original[mpeg4 526859B]  rerecorded[h264 1029884B]  동일=아님
000002  original[mpeg4 286879B]  rerecorded[h264  616068B]  동일=아님
000003  original[mpeg4 427410B]  rerecorded[h264  934964B]  동일=아님
000004  original[mpeg4 305244B]  rerecorded[h264  674416B]  동일=아님
000005  original[mpeg4 363811B]  rerecorded[h264  815289B]  동일=아님
```

이후 `catalog/integrity.json` 의 `stage1_class_collision` 판정은 `ok` 다.

### 남는 함정 — 코덱이 정답을 누설한다

복원된 상태에서 ORIGINAL 은 100% mpeg4, RERECORDED 는 100% h264 다. 컨테이너
메타데이터만 읽어도 라벨을 맞출 수 있다. 재녹화 예제가 실제 재촬영본이 아니라
**리샘플링·노이즈를 모사한 파생본**이라는 점과 합쳐지면, 공개 예제 5+5 로 학습한 모델은
비공개 평가에서 거의 확실히 무너진다.

실전 대응은 CCD 75,000장을 원본 풀로 삼아 재녹화 클래스를 직접 합성하고, 원본과
재녹화를 **동일한 코덱·비트레이트로 재인코딩**해 코덱 단서를 제거하는 것이다.
공개 예제는 코드 실행 확인용으로만 쓴다.

---

## 2. error — Stage 3 라벨 소실, 복구 불가

`data/stage3/labels.csv` 가 존재하지 않는다. 기존 문서 4곳
(`README.md`, `AGENTS.md`, `DATA_INDEX.md`, `data/data_index.json`)이
"25행, 0.1초 단위 희소 라벨" 이라고 참조하고 있었지만 파일 자체가 없었다.
배포본 `baseline.zip` 도 디스크에 남아 있지 않다.

**복구할 수 없다.** `OPEN_001`~`OPEN_005` 가 comma2k19 의 어느 세그먼트에서
나왔는지 알 방법이 없고, 로컬에 있는 comma2k19 예제 세그먼트 1개와는 내용이 다르다
(프레임 해시 대조로 확인). 정답을 추정해서 채우는 것은 잘못된 지표를 만들 뿐이므로
하지 않았다.

### 조치

- 파일을 만들지 않고 `error` 로 남겨 두었다. 조용히 넘어가는 것보다 낫다.
- 대체 경로로 `data/stage3/labels_comma2k19.csv` (600행)를 만들었다. CAN 정답에서
  파생한 **진짜 라벨**이고 스키마가 동일하다. 생성 방법은
  [04-datasets.md](04-datasets.md#5-활용--stage-3-라벨-생성) 참고.
- 원본 복원을 원하면 대회 페이지에서 `baseline.zip` 을 다시 받아
  `stage3/labels.csv` 만 `data/stage3/labels.csv` 로 복사하면 된다 (Dacon 로그인 필요).

---

## 3. 수정 완료: Stage 2 라벨 재생성

`data/stage2/labels.csv` 도 소실 상태였으나, 이쪽은 **결정론적으로 복원 가능**했다.
`data/SOURCES.md` 가 생성 규칙을 명시하기 때문이다 — "CCD 공식 `Crash-1500.txt` 주석의
첫 positive frame".

`data/external/Crash_Table.csv` 의 `frame_1..frame_50` 에서 첫 `1` 의 0-기반 인덱스를 취했다.

| ID | `t_collision` | 나머지 |
|---|---|---|
| 000001 | 32 | `-1` |
| 000002 | 30 | `-1` |
| 000003 | 31 | `-1` |
| 000004 | 41 | `-1` |
| 000005 | 30 | `-1` |

`t_entry`·`evasion_space`·`entry_side` 는 공개 정답이 없어 `-1` 이다(원 배포본도 동일).
재생성: `.venv/bin/python scripts/make_stage2_labels.py`

---

## 4. warning — Stage 3 프레임레이트가 대회 스펙과 다름

대회 스펙은 **10Hz** 이고 "1프레임 = 1 sample = 0.1초" 다. 그런데 공개 예제는 그렇지 않다.

| 파일 | 컨테이너 선언 | 실제 패킷 간격 | 디코딩 프레임 |
|---|---|---|---|
| OPEN_001.mp4 | 40/1 | 0.05s (**20Hz**) | 1200 |
| OPEN_002.mp4 | 40/1 | 0.05s | 1201 |
| OPEN_003.mp4 | 40/1 | 0.05s | 1197 |
| OPEN_004.mp4 | 40/1 | 0.05s | 1197 |
| OPEN_005.mp4 | 40/1 | 0.05s | 1197 |

comma2k19 원본이 20Hz × 60초 = 1,200 프레임인데(로컬 `video.hevc` 로 확인) 그대로
담긴 것으로 보인다. 문제가 셋이다.

1. **선언값 40fps 가 틀렸다.** 패킷 타임스탬프 간격은 0.05초, 즉 20Hz다.
2. **후반부 PTS 가 손상됐다.** 마지막 패킷들이 `2.501147, 2.501148, 2.501149` 로 1μs
   간격을 갖는다. 컨테이너 `duration` (2.5초)을 신뢰할 수 없다. 프레임 수는 반드시
   `-count_frames` 로 실측해야 한다.
3. **프레임 수가 일정하지 않다** (1197~1201).

### 영향

베이스라인 `predict_stage3` 는 디코딩된 프레임마다 `sample_index` 를 하나씩 발급한다.
입력이 10Hz 라는 가정이 깔려 있다. 20Hz 영상을 그대로 넣으면 sample 수가 2배가 되고
시간축이 어긋난다. 로컬에서 Stage 3 을 검증할 때는 **10Hz 로 리샘플링**하거나
`sample_index` 를 2프레임마다 발급하도록 맞춰야 한다.
`make_stage3_labels_from_comma2k19.py` 는 `--label-hz 10` 기준으로 `stride = round(20/10) = 2`
를 자동 계산한다.

---

## 5. warning — CrashBest 해상도 혼재

| 해상도 | 이미지 | 영상 |
|---|---|---|
| 1280×720 | 73,250 | 1,465 |
| 960×720 | 800 | 16 |
| 640×360 | 950 | 19 |

**영상 단위로는 일관**하다 — 한 영상 안에서 해상도가 섞이는 경우는 0건이다.
1,500개 중 35개만 비주류 해상도다.

주의할 점은 종횡비다. 1280×720 과 640×360 은 16:9 지만 **960×720 은 4:3** 이다.
베이스라인 전처리(`_crop_tensor`)는 짧은 변을 224 에 맞추고 중앙을 자르므로, 4:3
영상에서는 다른 화각이 잘려 나간다. 게다가 1280×720 이미지 상당수는 원본에
**필러박스(좌우 검은 띠)** 가 들어 있다 — 직접 확인한 `C_000001_01.jpg` 가 그렇다.
검은 띠는 실효 해상도를 낮추고, 하필 재녹화 판별에서 화면 재촬영의 특징으로 오인될
수 있는 종류의 신호다. 전처리 단계에서 검은 여백을 잘라내는(letterbox 제거) 처리를
넣을지 결정해야 한다.

---

## 6. info — CrashBest 연속 프레임 중복 (정상, 삭제 금지)

동일 내용 이미지 8장이 6그룹으로 존재한다.

| 그룹 | 파일 |
|---|---|
| 1 | `C_000676_44.jpg`, `C_000676_45.jpg` |
| 2 | `C_000826_43.jpg`, `C_000826_44.jpg` |
| 3 | `C_000826_49.jpg`, `C_000826_50.jpg` |
| 4 | `C_000989_43.jpg` … `C_000989_46.jpg` (4장) |
| 5 | `C_001173_41.jpg`, `C_001173_42.jpg` |
| 6 | `C_001173_43.jpg`, `C_001173_44.jpg` |

모두 **같은 영상의 연속 프레임**이다. 충돌 직후 장면이 정지한 구간에서 원본 영상이
같은 프레임을 반복한 결과다. 저장 낭비가 아니고, 삭제하면 "영상당 50프레임" 계약이
깨져 `Crash_Table.csv` 의 `frame_1..frame_50` 주석과 인덱스가 어긋난다. **유지한다.**

> 병렬 해싱 함정: `ls | xargs -P 8 -n 200 md5 -r` 로 75,000개를 해싱하면 프로세스
> 출력이 섞여 해시 문자열이 잘리고, 잘린 접두사가 우연히 겹쳐 **가짜 중복**이 나온다.
> 첫 점검에서 "e4b" 라는 17장 중복 그룹이 그렇게 만들어졌고, 개별 재검증에서 전부
> 서로 다른 파일임을 확인했다. `build_catalog.py` 는 파이썬 스레드 풀에서 해시를
> 계산해 이 문제가 없다.

---

## 7. ok — 의도된 중복 8그룹

카탈로그는 중복을 "의도된 것"과 "예상 밖"으로 나눈다. 현재 예상 밖 중복은 **0건**이다.

| 그룹 수 | 내용 | 사유 |
|---|---|---|
| 5 | `data/stage1/videos/original/*.mp4` ≡ `data/stage2/videos/*.mp4` | 설계상 같은 CCD 원본 클립 (`data/SOURCES.md`) |
| 3 | comma2k19 센서들의 `…/t` 배열 | 같은 클럭으로 샘플링돼 타임스탬프 배열을 공유 |

comma2k19 쪽 3그룹은 상위 데이터셋 포맷 자체의 특성이다.

- `CAN/speed/t` ≡ `CAN/wheel_speed/t`
- `IMU/accelerometer/t` ≡ `IMU/gyro/t` ≡ `IMU/gyro_bias/t` ≡ `IMU/gyro_uncalibrated/t`
- `IMU/magnetometer/t` ≡ `IMU/magnetometer_uncalibrated/t`

---

## 8. ok — 그 밖에 확인된 계약

| 항목 | 결과 |
|---|---|
| CrashBest 파일 수 | 75,000 = 1,500 × 50, 결측 0 |
| 영상당 프레임 수 | 전 영상 정확히 50 |
| `Crash_Table.csv` 조인 커버리지 | 1,500/1,500 (100.00%) |
| `Crash_Table.csv` 행 수 | 1,500 |
| Stage 1 클래스 충돌 | 없음 (복구 후) |
| Stage 1/2 라벨의 `path` 참조 | 전부 실제 파일로 연결 |
| comma2k19 센서 배열 형상 | 전부 파싱 성공, 형상 정합 |
| 조향 부호 규약 | ENU yaw rate 상관 −0.374 로 실측 검증 |

---

## 9. 평가 레이아웃 미러링 (구조 수정)

점검 중 발견한 별개 문제. 로컬 데이터가 `samples/stage1|2|3/` 에 있었는데
`baseline/baseline_train.ipynb` 는 `DATA = ROOT/'data'` 를 읽는다. 경로가 어긋나
베이스라인이 그대로 돌지 않았다. 또 Stage 2 는 평가 시 **영상이 아니라 프레임 이미지
폴더**를 입력으로 받는데 로컬에는 MP4 만 있었다.

`data/` 를 평가 서버 레이아웃과 일치시켰다.

| Stage | 평가 서버 입력 | 로컬 |
|---|---|---|
| 1 | `data_dir/videos/**` 재귀 탐색, ID = 파일 stem | `data/stage1/videos/{original,rerecorded}/` |
| 2 | `data_dir/images/<ID>/frame_XXXXXX.jpg`, ID = 폴더명 | `data/stage2/images/<ID>/` (`make stage2-images`) |
| 3 | `data_dir/videos/*.mp4`, ID = 파일 stem | `data/stage3/videos/` |

Stage 2 프레임 번호는 0-기반(`frame_000000`~`frame_000049`)으로 생성했다. 덕분에
`labels.csv` 의 `t_collision` 값이 제출 규격의 `collision_frame` 과 같은 좌표계를 갖는다.
평가 서버는 "파일명에 포함된 번호를 원본 프레임 번호로 그대로 사용"하므로, 이미지
순번을 새로 매기지 않도록 주의해야 한다.

---

## 10. 베이스라인 노트북의 알려진 결함

`baseline/baseline_inference.ipynb` 의 `inference.py` 생성 셀은 노트북 파일명을
하드코딩한다.

```python
NOTEBOOK_PATH = ROOT / "[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb"
```

로컬 파일명은 `baseline_inference.ipynb` 이므로 이 셀은 `FileNotFoundError` 로 죽는다.
셀을 실행하려면 `NOTEBOOK_PATH` 를 실제 파일명으로 바꾸거나 노트북을 원래 이름으로
저장해야 한다. 원본 배포 코드를 보존하기 위해 노트북 자체는 수정하지 않았다.

---

## 11. 문서에서 정정한 오류

이전 문서 세트에 있던 사실 오류를 바로잡았다.

| 위치 | 잘못된 내용 | 실제 |
|---|---|---|
| `README.md`, `data_index.json` | CrashBest 이미지가 128×128 | **1280×720** 외 2종 |
| `data_index.json` | Stage 1 ORIGINAL 파일 크기 | Stage 2 영상 크기를 잘못 기록 |
| `README.md` | `docs/`, `scripts/` 디렉터리 존재 | 당시 없었음 (지금은 생성됨) |
| `README.md` | 깨진 문자 `任务`, `멀티태스ك` | 각각 "과업", "멀티태스크" |
| `DATA_INDEX.md` | Stage 2·3 `labels.csv` 존재 | 파일 없음 |
| `AGENTS.md` | "baseline sample: 10개 영상 (5원본+5재녹화)" | 실제로는 동일 파일 5개의 중복 (복구 완료) |

중복 문서 `DATA_INDEX.md`, `data/data_index.json`, `data/manifest.json`, `dacon_guide.md` 는
[`catalog/catalog.json`](../catalog/catalog.json) 과 `docs/` 로 통합하고 삭제했다.
같은 사실이 여러 파일에 흩어져 서로 어긋나는 것이 위 오류들의 원인이었다.
