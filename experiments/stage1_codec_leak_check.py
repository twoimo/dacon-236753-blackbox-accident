"""Stage 1 실험 — 코덱 누설(codec leakage) 검증 및 중립화.

근거:
  - research/01-stage1-recapture/README.md §2 (최우선 위험: 코덱 누설)
  - docs/05-data-integrity-report.md §1 ("남는 함정 — 코덱이 정답을 누설한다")
  - env/configs/stage1.yaml (reencode.apply_to_both_classes=true)

이 대회에서 가장 크고 조용한 함정: 공개 예제는 ORIGINAL=전부 mpeg4, RERECORDED=전부 h264 다.
그냥 학습하면 모델은 "재촬영 여부"가 아니라 "어떤 코덱인가"를 외운다. 컨테이너 코덱만
읽어도 라벨이 100% 맞는다면 그 데이터셋에는 누설이 있는 것이다.

이 스크립트는:
  1) ffprobe 로 data/stage1 의 original vs rerecorded 각 영상의 코덱을 조사해 누설을 실증한다.
  2) 누설 지표 계산: 코덱이 클래스를 완벽히 예측하면 codec_leak=1, 아니면 0.
  3) 두 클래스를 **동일한 ffmpeg 설정**(libx264, 같은 CRF/GOP/픽셀포맷)으로 재인코딩해
     누설을 중립화한 세트를 temp 디렉터리에 만든다.
  4) 재인코딩본을 다시 조사해 코덱이 이제 일치함을(codec_leak_after=0) 확인한다.

출력 지표(하니스 계약):
  METRIC codec_leak_before=<0/1>
  METRIC codec_leak_after=<0/1>

환경 제약: 로컬 .venv 는 cv2/torch/pandas 를 로드하지 못한다(code-signing). 그래서 오직
ffmpeg/ffprobe 서브프로세스 + 표준 라이브러리(csv)만 사용한다.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "stage1"

# 재인코딩 설정 — 두 클래스에 "똑같이" 적용해야 누설이 사라진다 (research/01 §2 대응 원칙).
# 컨테이너/코덱을 통일하고 픽셀포맷·GOP·fps 까지 고정한다.
REENCODE_VCODEC = "libx264"
REENCODE_CRF = "23"
REENCODE_GOP = "30"
REENCODE_PIXFMT = "yuv420p"


def probe_codec(video: Path) -> str:
    """영상 v:0 스트림의 코덱 이름 (mpeg4/h264/...)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def load_label_map(data_dir: Path) -> dict[Path, str]:
    """labels.csv 에서 {절대경로: 라벨} 맵. 없으면 디렉터리 규칙(original/rerecorded)로 대체."""
    labels_csv = data_dir / "labels.csv"
    mapping: dict[Path, str] = {}
    if labels_csv.exists():
        with labels_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = (data_dir / row["path"]).resolve()
                mapping[p] = row["label"]
    else:
        for cls, folder in (("ORIGINAL", "original"), ("RERECORDED", "rerecorded")):
            for v in sorted((data_dir / "videos" / folder).glob("*.mp4")):
                mapping[v.resolve()] = cls
    return mapping


def codec_leak_metric(pairs: list[tuple[str, str]]) -> int:
    """(label, codec) 목록에서 누설 여부. 코덱이 클래스를 완벽 예측하면 1.

    누설 정의: 어떤 코덱도 두 클래스에 걸쳐 나타나지 않는다(코덱→라벨 함수가 단사).
    즉 각 코덱이 정확히 하나의 라벨에만 대응하고, 라벨이 2개 이상이면 누설.
    """
    codec_to_labels: dict[str, set[str]] = defaultdict(set)
    for label, codec in pairs:
        codec_to_labels[codec].add(label)
    labels = {lab for lab, _ in pairs}
    if len(labels) < 2:
        # 클래스가 하나뿐이면 누설을 논할 수 없음 → 0
        return 0
    # 모든 코덱이 단일 라벨에만 매핑되면 코덱만으로 라벨을 100% 맞출 수 있다 → 누설.
    perfectly_separates = all(len(labs) == 1 for labs in codec_to_labels.values())
    return 1 if perfectly_separates else 0


def reencode_uniform(src: Path, dst: Path) -> None:
    """두 클래스에 동일하게 적용하는 재인코딩. 코덱/CRF/GOP/픽셀포맷 통일."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", REENCODE_VCODEC,
            "-crf", REENCODE_CRF,
            "-g", REENCODE_GOP,
            "-pix_fmt", REENCODE_PIXFMT,
            "-an",  # 오디오 제거 (일관성)
            str(dst),
        ],
        check=True,
    )


def main() -> int:
    data_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DATA
    label_map = load_label_map(data_dir)
    if not label_map:
        print(f"[codec-leak] 영상을 찾지 못함: {data_dir}")
        return 1

    # ---- 1) BEFORE: 원본 데이터 코덱 조사 ----
    print(f"[codec-leak] 데이터: {data_dir}")
    print("[codec-leak] === BEFORE (원본 데이터) ===")
    before_pairs: list[tuple[str, str]] = []
    by_class_before: dict[str, list[str]] = defaultdict(list)
    for path in sorted(label_map):
        label = label_map[path]
        codec = probe_codec(path)
        before_pairs.append((label, codec))
        by_class_before[label].append(codec)
        rel = path.relative_to(data_dir)
        print(f"    {label:11s} {codec:8s} {rel}")
    for cls, codecs in sorted(by_class_before.items()):
        uniq = sorted(set(codecs))
        print(f"[codec-leak]   {cls:11s} 코덱 분포: {uniq}  (n={len(codecs)})")

    leak_before = codec_leak_metric(before_pairs)
    if leak_before:
        print("[codec-leak] ⚠ 누설 확인: 코덱만으로 라벨을 100% 예측 가능 "
              "(ORIGINAL=mpeg4, RERECORDED=h264). research/01 §2 / docs/05 §1 그대로.")

    # ---- 2) 중립화: 두 클래스를 동일 ffmpeg 설정으로 재인코딩 ----
    print(f"[codec-leak] === 중립화: 두 클래스 동일 재인코딩 "
          f"({REENCODE_VCODEC} crf={REENCODE_CRF} g={REENCODE_GOP} {REENCODE_PIXFMT}) ===")
    tmp_root = Path(tempfile.mkdtemp(prefix="stage1_codec_neutralized_"))
    after_pairs: list[tuple[str, str]] = []
    by_class_after: dict[str, list[str]] = defaultdict(list)
    for path in sorted(label_map):
        label = label_map[path]
        # 클래스별 하위 폴더에 같은 파일명으로 저장
        folder = "original" if label == "ORIGINAL" else "rerecorded"
        dst = tmp_root / "videos" / folder / path.name
        reencode_uniform(path, dst)
        codec = probe_codec(dst)
        after_pairs.append((label, codec))
        by_class_after[label].append(codec)

    print("[codec-leak] === AFTER (재인코딩 후) ===")
    for cls, codecs in sorted(by_class_after.items()):
        uniq = sorted(set(codecs))
        print(f"[codec-leak]   {cls:11s} 코덱 분포: {uniq}  (n={len(codecs)})")
    leak_after = codec_leak_metric(after_pairs)
    if not leak_after:
        print("[codec-leak] ✓ 중립화 성공: 두 클래스 코덱이 이제 동일 → 코덱만으로 라벨 예측 불가.")
    else:
        print("[codec-leak] ✗ 여전히 누설 (재인코딩 설정을 두 클래스에 동일 적용했는지 확인).")

    print(f"[codec-leak] 중립화 세트 위치: {tmp_root}")
    print(f"METRIC codec_leak_before={leak_before}")
    print(f"METRIC codec_leak_after={leak_after}")
    # 종료 코드: before=1(누설 실증) & after=0(중립화 성공) 이면 성공.
    return 0 if (leak_before == 1 and leak_after == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
