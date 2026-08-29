#!/usr/bin/env python3
"""Stage 2 학습용 특징 추출 — CCD 1,500영상 75k프레임 → npz 캐시.

학습형 충돌 국소화 헤드 학습을 빠르게 반복하기 위해, 각 프레임의 저차원 특징을
미리 추출해 디스크에 저장한다. torch/cv2 로드 가능(보안 해제됨), 그러나 프레임 특징은
numpy/PIL 로도 충분 — 재현성/속도 위해 PIL+numpy 사용.

특징(프레임당): 다운스케일 그레이스케일 32x32 flatten(1024) + 모션 관련 요약.
실제 학습 스크립트(stage2_train_head.py)가 이 npz 를 읽어 시퀀스 모델을 학습한다.

출력: experiments/cache/stage2_feats.npz
  X: (n_videos, max_frames, F) 프레임 특징
  motion: (n_videos, max_frames) 프레임간 모션 신호
  lengths: (n_videos,) 실제 프레임 수
  y: (n_videos,) first_crash_frame_index (0-based)
  vids: (n_videos,) 영상 ID
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CCD = REPO / "data" / "external" / "CrashBest"
CATALOG = REPO / "catalog" / "crashbest_videos.csv"
CACHE = REPO / "experiments" / "cache"
GRID = 32          # 프레임 그레이스케일 다운스케일 (특징 = GRID*GRID)
MAXF = 50          # CCD 는 전부 50프레임


def load_gray(p: Path, g: int = GRID) -> np.ndarray:
    im = Image.open(p).convert("L").resize((g, g), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    CACHE.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    F = GRID * GRID
    X = np.zeros((n, MAXF, F), dtype=np.float32)
    motion = np.zeros((n, MAXF), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    y = np.zeros(n, dtype=np.int32)
    vids = []
    t0 = time.time()
    ok = 0

    for i, r in enumerate(rows):
        vid = r["video_id"]
        fc = int(r["frame_count"])
        gt = int(r["first_crash_frame_index"])
        frames = []
        for k in range(1, fc + 1):
            p = CCD / f"C_{vid}_{k:02d}.jpg"
            if not p.exists():
                frames = []
                break
            frames.append(load_gray(p))
        if not frames:
            vids.append(vid)
            continue
        arr = np.stack(frames)                    # (fc, G, G)
        L = min(fc, MAXF)
        X[i, :L] = arr[:L].reshape(L, -1)
        # 프레임간 모션 (절대차 평균), index 0 = 0
        d = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
        motion[i, 1:L] = d[:L - 1]
        lengths[i] = L
        y[i] = gt
        vids.append(vid)
        ok += 1
        if ok % 200 == 0:
            print(f"  ... {ok}/{n} ({time.time()-t0:.0f}s)", flush=True)

    out = CACHE / "stage2_feats.npz"
    np.savez_compressed(out, X=X, motion=motion, lengths=lengths, y=y,
                        vids=np.array(vids))
    print(f"[extract] saved {out} ({out.stat().st_size/1e6:.1f}MB), ok={ok}/{n}, {time.time()-t0:.0f}s")
    print(f"METRIC stage2_feat_videos={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
