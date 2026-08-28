#!/usr/bin/env python3
"""Stage 2 평가서버 입력 레이아웃(프레임 이미지 폴더) 생성.

평가 환경의 predict_stage2(data_dir, model_dir) 는 영상이 아니라
`data_dir/images/<ID>/frame_XXXXXX.jpg` 형태의 프레임 이미지 폴더를 읽는다
(baseline_inference.ipynb 의 predict_stage2 참고). 배포된 공개 예제는 MP4 만
포함하므로, 로컬에서 추론 코드를 전 구간 검증하려면 프레임을 미리 추출해야 한다.

프레임 번호는 0-기반이다. 따라서 labels.csv 의 t_collision 값이 그대로
평가 제출값 collision_frame 과 같은 좌표계를 갖는다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def extract(video: Path, out_dir: Path, quality: int) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # -start_number 0 으로 frame_000000.jpg 부터 생성한다.
    command = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(video),
        "-fps_mode", "passthrough",
        "-q:v", str(quality),
        "-start_number", "0",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    return len(list(out_dir.glob("*.jpg")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v (2=최고품질)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg 가 필요합니다. `brew install ffmpeg`")

    videos_dir = args.root / "data" / "stage2" / "videos"
    images_dir = args.root / "data" / "stage2" / "images"
    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"영상이 없습니다: {videos_dir}")

    total = 0
    for video in videos:
        count = extract(video, images_dir / video.stem, args.quality)
        total += count
        print(f"  {video.stem}: {count} frames")
    print(f"생성 완료: {images_dir.relative_to(args.root)} (총 {total} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
