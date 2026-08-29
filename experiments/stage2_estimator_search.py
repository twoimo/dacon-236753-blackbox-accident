#!/usr/bin/env python3
"""Stage 2 충돌 국소화 추정기 개선 탐색 (numpy+PIL, CCD 서브셋).

기본 windowed_argmax(전역 모션피크)의 한계(ego:No 약함)를 개선할 변형들을 비교한다.
변형:
  - window_argmax: 기존 (prior 구간 평활 argmax)
  - onset: prior 구간에서 모션이 급증(1차차분 최대)하는 지점 = 충돌 시작
  - center: 중앙영역 가중 모션(주변부 카메라 흔들림 억제)
  - center_onset: 중앙가중 + onset
  - late_peak: prior 구간에서 임계 초과 첫 지점(peak 아니라 상승 시작)

사용: .venv/bin/python -m experiments.stage2_estimator_search [N]
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
FPS = 10.0
IMG = 96
PRIOR = 0.55


def load_stack(vid, fc):
    a = []
    for n1 in range(1, fc + 1):
        p = CCD / f"C_{vid}_{n1:02d}.jpg"
        if not p.exists():
            return None
        a.append(np.asarray(Image.open(p).convert("L").resize((IMG, IMG)), np.float32))
    return np.stack(a)


def mov(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same") if k > 1 else x


def global_motion(stack):
    d = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], d])


def center_motion(stack):
    h = stack.shape[1]
    c0, c1 = h // 4, 3 * h // 4
    cs = stack[:, c0:c1, c0:c1]
    d = np.abs(np.diff(cs, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], d])


def est(fc, sig, kind):
    lo = int(PRIOR * fc)
    s = mov(sig, 3)
    if kind in ("window", "center"):
        seg = s[lo:]
        return lo + int(np.argmax(seg)) if len(seg) else fc // 2
    if kind in ("onset", "center_onset"):
        # prior 구간에서 모션 1차차분(상승 기울기) 최대 = 충돌 시작
        seg = s[lo:]
        if len(seg) < 2:
            return fc // 2
        grad = np.diff(seg)
        return lo + int(np.argmax(grad)) + 1
    if kind == "late_peak":
        # prior 구간 최대의 60% 임계를 처음 넘는 지점
        seg = s[lo:]
        if len(seg) == 0:
            return fc // 2
        thr = 0.6 * seg.max()
        idx = np.where(seg >= thr)[0]
        return lo + int(idx[0]) if len(idx) else lo + int(np.argmax(seg))
    raise ValueError(kind)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8")))[:n]
    variants = {
        "window": ("global", "window"),
        "onset": ("global", "onset"),
        "center": ("center", "center"),
        "center_onset": ("center", "center_onset"),
        "late_peak": ("global", "late_peak"),
    }
    err = {k: [] for k in variants}
    err_ego = {k: {"Yes": [], "No": []} for k in variants}
    t0 = time.time()
    nok = 0
    for r in rows:
        vid, fc, gt = r["video_id"], int(r["frame_count"]), int(r["first_crash_frame_index"])
        st = load_stack(vid, fc)
        if st is None or len(st) < 3:
            continue
        gsig = global_motion(st)
        csig = center_motion(st)
        for k, (sigkind, estkind) in variants.items():
            sig = gsig if sigkind == "global" else csig
            e = abs(est(fc, sig, estkind) - gt)
            err[k].append(e)
            err_ego[k][r.get("egoinvolve", "Yes")].append(e)
        nok += 1
    print(f"[search] n={nok} ({time.time()-t0:.0f}s)")
    best = min(variants, key=lambda k: np.mean(err[k]))
    for k in variants:
        e = np.array(err[k])
        ey, en = np.array(err_ego[k]["Yes"]), np.array(err_ego[k]["No"])
        star = " <== best" if k == best else ""
        print(f"  {k:14s}: MAE={e.mean():6.3f}f  ego_yes={ey.mean():.2f} ego_no={en.mean():.2f}"
              f"  w3={np.mean(e<=3):.2f}{star}")
    print(f"METRIC search_best_mae_frames={np.mean(err[best]):.4f}")
    print(f"METRIC search_best_variant_ok=1")
    print(f"best_variant={best}")


if __name__ == "__main__":
    main()
