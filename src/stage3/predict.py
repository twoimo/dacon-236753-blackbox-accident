"""Stage 3 추론 — predict_stage3(data_dir, model_dir) 계약.

근거: research/03-stage3-egomotion/README.md, docs/05(20→10Hz 함정), docs/03(제출).
입력: data_dir/videos/*.mp4 (파일 stem=ID).
반환: DataFrame[ID, sample_index, accel_label, steer_label]  (0.1초 단위).

접근: 학습 없는 optical-flow ego-motion 베이스라인 (경로 B).
  - 프레임 수를 ffprobe -count_frames 로 실측 → 20→10Hz stride-2 리샘플 (docs/05).
  - 순수 numpy LK 흐름으로 속도 프록시(수직 흐름)·조향 프록시(지평선 수평 흐름) 추출.
    src/stage3/flow.py 의 로직을 experiments/stage3_flow_eval.py 와 공유한다.
  - comma2k19 예제 세그먼트 실측: accel_acc≈0.64(다수결 0.51),
    steer_acc(moving)≈0.85(다수결 0.84). mean_acc≈0.75.
  - STOPPED 프레임도 steer_label 필수(docs/03) → 항상 채운다.

프레임 추출:
  - 평가 서버(GPU, cv2/av 사용 가능)에서는 cv2 로 빠르게 디코드(fast-path).
  - cv2 미탑재(로컬 macOS)면 ffmpeg PNG 추출로 폴백. 둘 다 실패하면
    안전 폴백(CONSTANT/STRAIGHT)으로 규격만 지킨다.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _count_frames(video) -> int:
    """ffprobe -count_frames 실측 (docs/05: 컨테이너 fps/duration 신뢰 금지)."""
    from src.common.runtime import count_frames

    return count_frames(video)


def _load_10hz_cv2(video, w, h) -> np.ndarray | None:
    """cv2 로 전 프레임 그레이스케일 디코드 → stride-2 (20→10Hz). fast-path."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    frames = []
    ok, frame = cap.read()
    idx = 0
    while ok:
        if idx % 2 == 0:  # 20→10Hz stride-2
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(g.astype(np.float64))
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return np.stack(frames) if frames else None


def _load_10hz_ffmpeg(video, w, h) -> np.ndarray | None:
    """ffmpeg 로 전 프레임 그레이스케일 PNG 추출 → stride-2 (20→10Hz). 폴백."""
    from PIL import Image

    from src.stage3 import flow as F

    try:
        n_src = _count_frames(video)
    except Exception:
        n_src = 0
    with tempfile.TemporaryDirectory(prefix="stage3pred_") as td:
        try:
            files = F.extract_frames_ffmpeg(video, td, w=w, h=h)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        if not files:
            return None
        if n_src and len(files) != n_src:
            files = files[:n_src]
        sel = files[0::2]  # 10Hz
        if not sel:
            return None
        return np.stack([np.asarray(Image.open(f), dtype=np.float64) for f in sel])


def _load_10hz(video) -> np.ndarray | None:
    """10Hz 그레이스케일 프레임 (N,H,W). cv2 우선, 실패 시 ffmpeg."""
    from src.stage3 import flow as F

    imgs = _load_10hz_cv2(video, F.EXTRACT_W, F.EXTRACT_H)
    if imgs is None:
        imgs = _load_10hz_ffmpeg(video, F.EXTRACT_W, F.EXTRACT_H)
    return imgs


def predict_stage3(data_dir, model_dir):
    import pandas as pd

    from src.common import data as D
    from src.common.runtime import resample_indices
    from src.common.submit_guard import check_stage3
    from src.stage3 import flow as F

    videos = D.stage3_videos(data_dir)
    _ = Path(model_dir) / "best.pt"  # 학습 없는 베이스라인: 가중치 없어도 동작

    rows = []
    for v in videos:
        vid = v.stem
        # docs/05: fps 신뢰 금지 → 실측 후 20Hz 가정으로 10Hz 리샘플 길이 결정.
        try:
            n_src = _count_frames(v)
        except Exception:
            n_src = 0
        n_samples = len(resample_indices(n_src, src_hz=20.0, target_hz=10.0)) if n_src else 0

        imgs = _load_10hz(v)
        if imgs is not None and len(imgs) >= 3:
            speed, u_med = F.compute_flow_features(imgs)
            accel = F.classify_accel(speed)
            steer = F.classify_steer(u_med)  # STOPPED 여부와 무관히 항상 채움 (docs/03)
            n_out = n_samples if n_samples else len(accel)
            if len(accel) != n_out:  # 리샘플 길이와 정합 보정
                m = min(len(accel), n_out)
                accel = list(accel[:m]) + ["CONSTANT"] * (n_out - m)
                steer = list(steer[:m]) + ["STRAIGHT"] * (n_out - m)
        else:
            # 디코드 완전 실패: 규격만 지키는 안전 폴백.
            n_out = n_samples if n_samples else 1
            accel = ["CONSTANT"] * n_out
            steer = ["STRAIGHT"] * n_out

        for si in range(len(accel)):
            rows.append({
                "ID": vid,
                "sample_index": si,
                "accel_label": accel[si],
                "steer_label": steer[si],
            })

    cols = ["ID", "sample_index", "accel_label", "steer_label"]
    df = pd.DataFrame(rows, columns=cols)
    return check_stage3(df)


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage3"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage3"
    out = predict_stage3(dd, md)
    print(out.head(10).to_string(index=False))
    print(f"METRIC stage3_rows={len(out)}")
