#!/usr/bin/env python3
"""Stage 3 학습용 특징 추출 — comma2k19 영상 → 프레임별 흐름/외형 특징 npz 캐시.

학습형 가감속·조향 분류기(stage3_train_head.py)용 입력을 만든다. comma2k19 example 영상은
실제 20Hz 이므로 10Hz(매 2프레임)로 리샘플해 라벨(600행)과 sample_index 로 정렬한다
(docs/05 프레임률 함정 대응).

특징(프레임당 72차원):
  - 전역 LK 흐름 (dx, dy): 수직=속도 프록시, 수평=요/조향
  - 지평선 밴드 열별 수평 shift 4개 + 그 중앙값 (요 신호 강건화)
  - 프레임간 절대차 평균 (모션 크기)
  - 8x8 다운스케일 외형 (64차원)

출력: experiments/cache/stage3_feats.npz (X, accel_y, steer_y, sample_index)
실행: .venv/bin/python -m experiments.stage3_extract_features
"""
from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data/external/comma2k19/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/video.hevc"
LABELS = REPO / "data/stage3/labels_comma2k19.csv"
CACHE = REPO / "experiments" / "cache"
W, H = 96, 72          # 흐름 계산 해상도
GRID = 8               # 외형 다운스케일 (8x8=64차원)
# 라벨 클래스 인덱스 (stage3_train_head.py 의 ACCEL_NAMES/STEER_NAMES 와 일치)
ACCEL_CLS = {"STOPPED": 0, "DECELERATING": 1, "CONSTANT": 2, "ACCELERATING": 3}
STEER_CLS = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}


def extract_frames_10hz(video: Path, out_dir: Path) -> list[Path]:
    """20Hz 영상 → 10Hz grayscale PNG. -fps_mode passthrough 로 PTS 손상 대응(docs/05)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"select='not(mod(n,2))',scale={W}:{H},format=gray",
           "-fps_mode", "passthrough", str(out_dir / "f_%04d.png")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("f_*.png"))


def lk_region(a: np.ndarray, b: np.ndarray, y0: int, y1: int, x0: int, x1: int):
    """Lucas-Kanade 전역 이동 추정 → (dx, dy)."""
    ar = a[y0:y1, x0:x1]; br = b[y0:y1, x0:x1]
    if ar.size == 0:
        return 0.0, 0.0
    ix = np.zeros_like(ar); iy = np.zeros_like(ar)
    ix[:, 1:-1] = (ar[:, 2:] - ar[:, :-2]) * 0.5
    iy[1:-1, :] = (ar[2:, :] - ar[:-2, :]) * 0.5
    it = br - ar
    A = np.array([[np.sum(ix * ix), np.sum(ix * iy)],
                  [np.sum(ix * iy), np.sum(iy * iy)]])
    rhs = -np.array([np.sum(ix * it), np.sum(iy * it)])
    try:
        v = np.linalg.solve(A + np.eye(2) * 1e-3, rhs)
        return float(v[0]), float(v[1])
    except Exception:
        return 0.0, 0.0


def frame_features(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    h, w = cur.shape
    feats: list[float] = []
    # 전역 흐름 (하단 3/4 = 도로면 위주, 하늘 제외)
    dx, dy = lk_region(prev, cur, h // 4, h, 0, w)
    feats += [dx, dy]
    # 지평선 밴드 열별 수평 shift (요/조향 신호)
    y0, y1 = int(0.25 * h), int(0.5 * h)
    step = max(1, w // 4)
    strips = [lk_region(prev, cur, y0, y1, xs, min(xs + step, w))[0]
              for xs in range(0, w, step)]
    strips = strips[:4] + [0.0] * max(0, 4 - len(strips))
    feats += strips[:4]
    feats.append(float(np.median(strips[:4])))
    # 모션 크기
    feats.append(float(np.abs(cur - prev).mean()))
    # 외형 다운스케일
    small = np.asarray(Image.fromarray(cur.astype(np.uint8)).resize((GRID, GRID)),
                       dtype=np.float32) / 255.0
    feats += small.reshape(-1).tolist()
    return np.array(feats, dtype=np.float32)


def main() -> int:
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
    n = len(rows)
    print(f"[s3-extract] labels={n}, 10Hz 프레임 추출 중...")
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames_10hz(VIDEO, Path(td))
        print(f"[s3-extract] 프레임 {len(frames)}개 (라벨 {n})")
        grays = [np.asarray(Image.open(p), dtype=np.float32) for p in frames]
        m = min(len(grays), n)
        X = np.stack([frame_features(grays[i - 1] if i > 0 else grays[i], grays[i])
                      for i in range(m)])
    accel_y = np.array([ACCEL_CLS[rows[i]["accel_label"]] for i in range(m)], dtype=np.int64)
    steer_y = np.array([STEER_CLS[rows[i]["steer_label"]] for i in range(m)], dtype=np.int64)
    sidx = np.array([int(rows[i]["sample_index"]) for i in range(m)], dtype=np.int64)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "stage3_feats.npz"
    np.savez_compressed(out, X=X, accel_y=accel_y, steer_y=steer_y, sample_index=sidx)
    print(f"[s3-extract] saved {out}: X={X.shape} "
          f"accel={np.bincount(accel_y, minlength=4)} steer={np.bincount(steer_y, minlength=3)}")
    print(f"METRIC stage3_feat_samples={len(X)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
