# 협업 가이드

여러 사람이 각자 컴퓨터에서 브랜치나 워크트리를 파서 동시에 작업하는 것을 전제로 한다.
핵심 제약은 **데이터가 깃에 없다**는 점이다. 8GB 를 저장소에 넣을 수 없으므로 코드와
데이터의 생애주기가 분리된다. 이 문서는 그 분리를 어떻게 다루는지 설명한다.

---

## 1. 처음 셋업

```bash
git clone https://github.com/twoimo/dacon-236753-blackbox-accident.git
cd dacon-236753-blackbox-accident

make setup          # .venv 구성 (python3.11 + numpy + pillow)
make data           # 구글 드라이브 공개 링크에서 대용량 데이터 복원 (~8.3GB)
make stage2-images  # Stage2 평가 레이아웃 프레임 이미지 생성
make verify         # 카탈로그와 대조 검증
```

`make data` 는 구글 계정도 `rclone` 설정도 필요 없다. 공개 링크로만 동작한다.
링크 목록은 [07-dataset-distribution.md](07-dataset-distribution.md) 에 있다.

필요한 도구: `git`, `python3.11+`, `ffmpeg`, `zstd`, `curl`.

```bash
brew install ffmpeg zstd        # macOS
sudo apt install ffmpeg zstd    # Debian/Ubuntu
```

---

## 2. 데이터를 한 번만 내려받고 워크트리끼리 공유하기

여기가 이 저장소에서 제일 중요한 운영 요령이다.

워크트리를 3개 만들면 기본적으로 데이터도 3벌, 즉 25GB 를 쓰게 된다. 그럴 필요가 없다.
데이터를 저장소 **바깥** 한 곳에 두고 각 워크트리에서 심볼릭 링크로 가리키면 된다.

```bash
# 1) 공유 데이터 저장소를 저장소 바깥에 만든다
mkdir -p ~/dacon-shared

# 2) 주 클론에서 데이터를 한 번만 받는다
cd ~/repos/dacon-236753-blackbox-accident
make data
mv data/external          ~/dacon-shared/external
mv data/stage3/videos     ~/dacon-shared/stage3-videos

# 3) 링크로 되돌린다
ln -s ~/dacon-shared/external      data/external
ln -s ~/dacon-shared/stage3-videos data/stage3/videos
```

새 워크트리를 만들 때마다 링크만 다시 걸면 된다.

```bash
git worktree add ../wt-stage1 feat/stage1-augmentation
cd ../wt-stage1
ln -s ~/dacon-shared/external      data/external
ln -s ~/dacon-shared/stage3-videos data/stage3/videos
make setup && make stage2-images
```

`data/external` 과 `data/stage3/videos` 는 `.gitignore` 에 있으므로 심볼릭 링크가
커밋되지 않는다. `catalog/` 의 경로는 저장소 루트 기준 상대 경로라서 링크 여부와
무관하게 `make verify` 가 그대로 동작한다.

`data/stage2/images/` 는 링크하지 말고 워크트리별로 생성하는 편이 낫다. 12MB 밖에
안 되고 `make stage2-images` 가 몇 초에 끝난다.

---

## 3. 워크트리 운영

```bash
# 새 작업 시작
git worktree add ../wt-<주제> -b feat/<주제>

# 현황 확인
git worktree list

# 정리 (브랜치는 남는다)
git worktree remove ../wt-<주제>
git worktree prune
```

권장 배치:

```text
~/repos/
├── dacon-236753-blackbox-accident/   # main. 실험하지 않는다.
├── wt-stage1/                        # feat/stage1-…
├── wt-stage2/                        # feat/stage2-…
└── wt-stage3/                        # feat/stage3-…
```

`.venv` 는 워크트리마다 따로 만든다(`.gitignore` 대상). 워크트리 간 공유하면
경로가 하드코딩돼 깨진다.

---

## 4. 브랜치 규칙

| 접두사 | 용도 |
|---|---|
| `feat/stage1-…` `feat/stage2-…` `feat/stage3-…` | Stage별 모델·전처리 작업 |
| `data/…` | 데이터셋 추가·정제, 카탈로그 스키마 변경 |
| `exp/…` | 버려질 수 있는 실험. 머지하지 않아도 된다 |
| `fix/…` | 버그 수정 |
| `docs/…` | 문서만 변경 |

`main` 은 항상 `make check` 가 통과하는 상태로 유지한다. Stage 담당이 갈려 있으면
서로 다른 디렉터리를 만지므로 충돌이 거의 없다.

---

## 5. 생성물은 커밋하지 않는다 — 단 하나 예외

| 대상 | 커밋 | 이유 |
|---|---|---|
| `catalog/**` | **한다** | 데이터 없이도 AI·사람이 데이터셋을 조회할 수 있어야 한다. 이 저장소의 핵심 산출물이다. |
| `data/stage1/`, `data/stage2/videos/` | 한다 | 8MB. 클론 직후 바로 쓸 수 있는 편의가 용량보다 크다 |
| `data/external/`, `data/stage3/videos/` | 안 한다 | 8.3GB |
| `data/stage2/images/` | 안 한다 | 재생성이 몇 초 |
| `model/`, `output/`, `*.pt` | 안 한다 | 학습 산출물 |
| `.venv/` | 안 한다 | 환경 |
| `dist/` | 안 한다 | 배포 아카이브 |

### 카탈로그 충돌 다루기

`catalog/` 는 생성물이라 여러 브랜치에서 재생성하면 충돌한다. 수동으로 병합하지 말고
재생성한다.

```bash
git checkout --theirs catalog/ 2>/dev/null || true
make catalog
git add catalog/
```

`catalog/crashbest_index.csv` 는 75,000행이다. 데이터를 실제로 바꾸지 않았다면
`make catalog` 를 돌리지 않는 편이 diff 를 깨끗하게 유지한다. `.gitattributes` 에서
`catalog/**` 를 `linguist-generated=true` 로 표시해 두었으므로 GitHub PR 에서는
기본으로 접힌다.

---

## 6. 데이터를 바꿨을 때

데이터셋을 추가하거나 정제했다면 코드 변경과 같은 커밋에 카탈로그 갱신을 포함한다.

```bash
make catalog                                  # 카탈로그 + 무결성 판정 재생성
make verify                                   # 대조 검증
git add catalog/ docs/                        # 판정 근거는 docs/05 에 기록
git commit -m "data: <무엇을 어떻게 바꿨는지>"
```

공유 데이터 자체를 갱신해 다른 사람에게도 배포해야 한다면 새 버전으로 올린다.

```bash
DRIVE_DIR=dacon-236753-blackbox-accident/v2 make publish
```

`scripts/gdrive_upload.sh` 가 `catalog/distribution.json` 과
`docs/07-dataset-distribution.md` 를 갱신하므로, 그 두 파일을 커밋하면 다른 사람의
`make data` 가 새 버전을 받는다. 기존 `v1` 링크는 그대로 살아 있으니 급하게
갈아탈 필요는 없다.

---

## 7. 검증 명령

```bash
make check                    # 카탈로그 대조 + 무결성 요약 (평소 이것만)
make verify-full              # sha256 전수 대조 (느림, 데이터 이관 후)
make verify                   # 크기 기준 + CrashBest 2000장 표본
```

`verify_integrity.py` 는 불일치가 있으면 종료 코드 1 을 반환한다. CI 나 pre-push
훅에 걸어 쓸 수 있다.

---

## 8. 자주 겪는 문제

**`make data` 가 해시 불일치로 실패한다.**
구글 드라이브가 대용량 파일에 바이러스 검사 확인 페이지(HTML)를 돌려주고 그 HTML 이
파일로 저장된 경우다. [07-dataset-distribution.md](07-dataset-distribution.md) 의
공유 링크로 브라우저에서 직접 받아 `dist/` 에 두고 `make data` 를 다시 실행하면
스크립트가 해시를 확인하고 압축 해제만 수행한다.

**`make verify` 가 `data/stage3/labels.csv` 결측을 보고한다.**
알려진 상태다. 복구 불가 항목이며 이유는
[05-data-integrity-report.md](05-data-integrity-report.md) 2절에 있다.
대체 라벨은 `data/stage3/labels_comma2k19.csv` 다.

**베이스라인 추론이 `RuntimeError: CUDA GPU 평가환경을 필요로 합니다` 로 죽는다.**
`inference.py` 의 `_device()` 가 CUDA 를 강제한다. macOS 에서는 정상 동작이다.
전처리·후처리만 따로 검증하거나 CUDA 머신에서 돌린다.

**워크트리에서 `data/` 가 비어 있다.**
심볼릭 링크를 걸지 않았다. 2절 참고.
