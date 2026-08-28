# 에이전트 작업 규칙

DACON 236753 (블랙박스 영상 기반 지능형 고의사고 분석). 사람용 문서는 `docs/`,
기계 판독 정본은 `catalog/`.

## 사실을 확인하는 순서

1. **`catalog/catalog.json`** — 대회 메타, Stage 스키마, 데이터셋 요약. 수치의 정본.
2. **`catalog/*.csv`** — 파일·미디어·이미지 단위 인덱스.
3. **`catalog/integrity.json`** — 무결성 판정과 알려진 결함.
4. `docs/` — 위 내용의 사람용 서술.

카탈로그와 문서가 어긋나면 **카탈로그가 맞다.** `scripts/build_catalog.py` 가 실제
파일을 스캔해 생성한다. 문서를 근거로 수치를 주장하지 말고 카탈로그를 조회한다.

`data/external/` (8.1GB), `data/stage3/videos/` (179MB) 는 저장소에 없을 수 있다.
없을 때도 `catalog/` 만으로 데이터셋 구조·통계·라벨을 전부 조회할 수 있게 설계돼 있다.
파일이 실제로 필요할 때만 `make data` 를 안내한다.

## 하지 말아야 할 것

- **라벨을 추측해서 만들지 않는다.** `data/stage3/labels.csv` 는 소실되었고 복구
  불가다(근거: `docs/05-data-integrity-report.md` 2절). 그럴듯한 값으로 채우면
  지표가 조용히 망가진다. 대체는 `data/stage3/labels_comma2k19.csv`.
- **CrashBest 의 "중복" 이미지를 삭제하지 않는다.** 8장은 원본 영상 정지 구간의 연속
  프레임이다. 지우면 "영상당 50프레임" 계약이 깨져 `Crash_Table.csv` 주석 인덱스가 어긋난다.
- **`baseline/` 을 수정하지 않는다.** 대회 배포 원본이다. 알려진 결함이 있어도
  `docs/05` 에 기록만 한다.
- **`catalog/` 와 `docs/07-dataset-distribution.md` 를 손으로 편집하지 않는다.** 생성물이다.
- **`data/` 레이아웃을 바꾸지 않는다.** 평가 서버 구조와 1:1 이라서 베이스라인이 그대로 돈다.

## 데이터 레이아웃 (평가 서버와 동일)

| Stage | 평가 시 입력 | 로컬 경로 | ID |
|---|---|---|---|
| 1 | `data_dir/videos/**` 재귀 | `data/stage1/videos/{original,rerecorded}/` | 파일 stem |
| 2 | `data_dir/images/<ID>/frame_XXXXXX.jpg` | `data/stage2/images/<ID>/` | 폴더명 |
| 3 | `data_dir/videos/*.mp4` | `data/stage3/videos/` | 파일 stem |

**Stage 2 만 이미지 폴더 입력이다.** Stage 1·3 은 영상 파일이다. 가장 자주 틀리는 지점.
Stage 2 프레임 번호는 파일명 숫자를 그대로 원본 프레임 번호로 쓴다(재번호 금지).
로컬은 0-기반(`frame_000000`~)이라 `labels.csv` 의 `t_collision` 과 제출값
`collision_frame` 이 같은 좌표계다.

## 라벨·제출 스키마

정본은 `catalog/catalog.json` 의 `.stages`. 요약:

| Stage | labels.csv | 제출 CSV |
|---|---|---|
| 1 | `ID, path, label` — `ORIGINAL`/`RERECORDED` | `ID, answer` |
| 2 | `ID, path, t_collision, t_entry, evasion_space, entry_side` — `-1`=정답없음 | `ID, collision_frame, entry_frame, evasion_space, entry_side` |
| 3 | `ID, sample_index, frame_index, time_seconds, accel_label, steer_label` | `ID, sample_index, accel_label, steer_label` |

- `accel_label`: `ACCELERATING` / `DECELERATING` / `CONSTANT` / `STOPPED`
- `steer_label`: `LEFT` / `STRAIGHT` / `RIGHT`
- `evasion_space`: `0`=없음 / `1`=있음 · `entry_side`: `LEFT` / `RIGHT`
- Stage 3 `STOPPED` 프레임은 조향 채점에서 제외되지만 **제출에는 `steer_label` 이 있어야 한다.**

## 모델링 시 반드시 반영할 데이터 함정

근거는 전부 `docs/05-data-integrity-report.md`.

1. **Stage 1 코덱 누설** — 공개 예제 ORIGINAL 은 전부 mpeg4, RERECORDED 는 전부 h264.
   공개 예제로 학습하면 코덱을 배운다. CCD 75,000장으로 재녹화를 합성하고 두 클래스를
   같은 코덱·비트레이트로 재인코딩할 것.
2. **재녹화 예제는 진짜 재촬영본이 아니다** — 리샘플링·노이즈 모사 파생본.
3. **Stage 3 는 20Hz, 스펙은 10Hz** — 컨테이너는 40fps 로 오선언, 후반 PTS 손상으로
   `duration` 신뢰 불가. 프레임 수는 `ffprobe -count_frames` 로 실측. 프레임당
   `sample_index` 를 발급하면 시간축이 2배 어긋난다.
4. **CrashBest 해상도 3종** — 1280×720 73,250장 / 960×720 800장 / 640×360 950장.
   960×720 은 4:3 이라 중앙 크롭 시 화각이 달라진다. 1280×720 다수에 좌우 필러박스
   (검은 띠)가 있고, 이것이 재녹화 판별에서 거짓 단서가 될 수 있다.
5. **평가 서버 인터넷 차단** — 모든 모델은 `weights=None`, 가중치는 `model/` 에 포함.

## 명령

```bash
make setup            # .venv (python3.11 + numpy + pillow)
make data             # 구글 드라이브에서 대용량 데이터 복원
make labels           # Stage2 라벨 복원 + comma2k19 파생 라벨
make stage2-images    # Stage2 평가 레이아웃 프레임 생성
make catalog          # 카탈로그 + 무결성 재생성 (~1분)
make check            # 대조 검증 + 요약
```

데이터나 라벨을 변경했으면 **같은 커밋에 `make catalog` 결과를 포함한다.**

## 환경

- 도구용: `.venv` (python3.11, numpy, pillow) — `requirements-tools.txt`
- 학습/추론: `baseline/requirements.txt` (torch 2.8.0, torchvision 0.23.0, opencv 4.10.0.84)
- 외부 명령: `ffmpeg`/`ffprobe`, `zstd`, `curl`, `rclone`(배포 시), `gh`(깃허브)
- 베이스라인 추론은 CUDA 를 강제한다. macOS 에서는 `predict_*` 를 그대로 실행할 수 없다.
