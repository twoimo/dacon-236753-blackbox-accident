#!/usr/bin/env python3
"""Stage 2 충돌 국소화 — CCD 1,500영상 전체 대규모 실측 (numpy+PIL만).

기존 stage2_collision_eval.py 는 로컬 샘플 5개만 검증했다. 이 스크립트는 로컬에 있는
CrashBest 75,000 프레임(1,500영상)과 catalog/crashbest_videos.csv 의
first_crash_frame_index(0-based 정답)를 이용해 충돌 국소화 MAE 를 통계적으로 측정한다.

파일명 규약: data/external/CrashBest/C_<vid>_<NN>.jpg  (NN=01..50, 1-based)
정답: first_crash_frame_index (0-based) → 파일 프레임번호는 idx+1.

지표: METRIC ccd_collision_mae_frames / mae_sec (10fps 가정) + 방법/환경별 분석.
cv2/torch/pandas 불필요 (macOS 로컬 실행 가능).
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CCD_DIR = REPO / "data" / "external" / "CrashBest"
CATALOG = REPO / "catalog" / "crashbest_videos.csv"
FPS = 10.0
IMG = 96  # 다운스케일 (모션 신호엔 충분, 속도 우선)
CCD_MIN_FRAC = 0.55  # CCD prior: 충돌은 항상 클립 후반 (min 0.60, 여유 0.55)


def load_catalog() -> list[dict]:
    with open(CATALOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_path(vid: str, n1: int) -> Path:
    # n1: 1-based file number
    return CCD_DIR / f"C_{vid}_{n1:02d}.jpg"


def load_gray_stack(vid: str, frame_count: int) -> np.ndarray | None:
    arrs = []
    for n1 in range(1, frame_count + 1):
        p = frame_path(vid, n1)
        if not p.exists():
            return None
        try:
            im = Image.open(p).convert("L").resize((IMG, IMG))
        except Exception:
            return None
        arrs.append(np.asarray(im, dtype=np.float32))
    return np.stack(arrs) if arrs else None


def motion_signal(stack: np.ndarray) -> np.ndarray:
    """프레임 간 절대차 평균 → 길이 N (첫 프레임 0)."""
    d = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], d])


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def estimate(nums_n: int, sig: np.ndarray, method: str, smooth_k: int = 3) -> int:
    """0-based 충돌 프레임 인덱스 추정."""
    if method == "argmax":
        return int(np.argmax(sig))
    if method == "smooth_argmax":
        return int(np.argmax(moving_average(sig, smooth_k)))
    if method == "windowed_argmax":
        lo = int(CCD_MIN_FRAC * nums_n)
        seg = moving_average(sig, smooth_k)[lo:]
        return lo + int(np.argmax(seg)) if len(seg) else nums_n // 2
    if method == "midpoint":
        return nums_n // 2
    raise ValueError(method)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0=전체
    rows = load_catalog()
    if limit:
        rows = rows[:limit]

    methods = ["midpoint", "argmax", "smooth_argmax", "windowed_argmax"]
    abs_err: dict[str, list[float]] = defaultdict(list)
    env_err: dict[str, list[float]] = defaultdict(list)  # windowed 기준 환경별
    n_ok = 0
    n_skip = 0
    t0 = time.time()

    for i, r in enumerate(rows):
        vid = r["video_id"]
        fc = int(r["frame_count"])
        gt = int(r["first_crash_frame_index"])  # 0-based
        stack = load_gray_stack(vid, fc)
        if stack is None or len(stack) < 3:
            n_skip += 1
            continue
        sig = motion_signal(stack)
        for m in methods:
            pred = estimate(fc, sig, m)
            abs_err[m].append(abs(pred - gt))
        # 환경별 (windowed)
        pred_w = estimate(fc, sig, "windowed_argmax")
        env_err[f"timing:{r.get('timing','?')}"].append(abs(pred_w - gt))
        env_err[f"weather:{r.get('weather','?')}"].append(abs(pred_w - gt))
        env_err[f"ego:{r.get('egoinvolve','?')}"].append(abs(pred_w - gt))
        n_ok += 1
        if n_ok % 200 == 0:
            el = time.time() - t0
            print(f"  ... {n_ok} videos ({el:.0f}s)", flush=True)

    print(f"[ccd-eval] evaluated={n_ok} skipped={n_skip} ({time.time()-t0:.0f}s)")
    print(f"[ccd-eval] IMG={IMG} fps={FPS} prior_frac={CCD_MIN_FRAC}")
    print("\n=== 방법별 MAE (프레임 / 초) ===")
    best_m, best_v = None, 1e9
    for m in methods:
        e = np.array(abs_err[m])
        mae = e.mean()
        print(f"  {m:16s}: {mae:6.3f} frames  ({mae/FPS:.3f} s)   "
              f"med={np.median(e):.1f} p90={np.percentile(e,90):.1f}")
        if m != "midpoint" and mae < best_v:
            best_v, best_m = mae, m

    e_best = np.array(abs_err[best_m])
    e_base = np.array(abs_err["midpoint"])
    print(f"\n최적 방법: {best_m}")
    print("\n=== 환경별 MAE (windowed_argmax) ===")
    for k in sorted(env_err):
        e = np.array(env_err[k])
        print(f"  {k:18s}: {e.mean():6.3f} frames  (n={len(e)})")

    # 채점 관점: 초 단위 오차 (Stage2 지표)
    print("\n=== METRIC ===")
    print(f"METRIC ccd_n_eval={n_ok}")
    print(f"METRIC ccd_collision_mae_frames={e_best.mean():.4f}")
    print(f"METRIC ccd_collision_mae_sec={e_best.mean()/FPS:.4f}")
    print(f"METRIC ccd_baseline_mid_mae_frames={e_base.mean():.4f}")
    print(f"METRIC ccd_improvement_frames={e_base.mean()-e_best.mean():.4f}")
    print(f"METRIC ccd_within_1frame_rate={(e_best<=1).mean():.4f}")
    print(f"METRIC ccd_within_3frame_rate={(e_best<=3).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
