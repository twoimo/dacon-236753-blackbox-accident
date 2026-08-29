"""Stage 2 충돌 프레임 국소화 — 로컬 측정 실험 (numpy + PIL + csv only).

목적
----
research/02-stage2-anticipation/README.md 의 §3.1 "충돌 = 시간 국소화(peak detection)"
가설을, 로컬 5개 샘플 영상(data/stage2/images/<ID>/frame_XXXXXX.jpg)에서 **실측**한다.

왜 numpy+PIL+csv 만 쓰는가
------------------------
로컬 .venv 는 code-signing 정책 때문에 pandas/cv2/torch 의 확장 모듈(.so) 로드가
차단된다(검증됨). 따라서 로컬에서 돌릴 수 있는 측정 코드는 표준 csv 모듈 + numpy + PIL
만으로 작성한다. 제출 경로(src/stage2/predict.py)의 무거운 의존은 try/except 로 가드한다.

추정기 (collision_frame estimator)
--------------------------------
프레임 간 전역 변화(global motion energy) 의 피크를 충돌 프레임으로 본다.
  1) 각 프레임을 그레이스케일 + 다운스케일(기본 96x96) 로 읽어 float 배열화.
  2) 인접 프레임 절대차의 평균 d[t] = mean(|f[t]-f[t-1]|), t=1..N-1.
     충돌은 화면 전체가 급격히 흔들리는/가려지는 순간이라 d 가 스파이크를 만든다.
  3) 선택 규칙 비교:
       - argmax      : d 최대 지점의 (뒤쪽) 프레임 t.
       - smooth_argmax: d 를 짧은 이동평균으로 평활 후 argmax (단발 노이즈 억제).
  4) 프레임 번호는 파일명 숫자 그대로 사용(재번호 금지). 로컬 0-기반이라
     labels.csv 의 t_collision 과 같은 좌표계.

지표
----
정답 t_collision 대비 프레임/초 MAE. 초 변환은 10fps 가정(CCD=50프레임/5초).
naive midpoint(N//2) 베이스라인과 비교한다.

출력 (하니스 계약)
------------------
  METRIC collision_mae_frames=<v>
  METRIC collision_mae_sec=<v>
  (+ 베이스라인/방법별 보조 라인)

실행:
  .venv/bin/python -m experiments.stage2_collision_eval
  .venv/bin/python -m experiments.stage2_collision_eval --data-dir data/stage2 --fps 10 --size 96
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# 데이터 로딩 (평가 레이아웃과 동일: data_dir/images/<ID>/frame_XXXXXX.jpg)
# ----------------------------------------------------------------------------
def list_id_dirs(data_dir: Path) -> list[Path]:
    root = data_dir / "images"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def list_frames(id_dir: Path) -> list[Path]:
    return sorted(id_dir.glob("frame_*.jpg"))


def frame_number(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def load_gray_stack(frames: list[Path], size: int) -> np.ndarray:
    """프레임들을 (N, size, size) float32 그레이스케일 스택으로."""
    arrs = []
    for f in frames:
        im = Image.open(f).convert("L").resize((size, size), Image.BILINEAR)
        arrs.append(np.asarray(im, dtype=np.float32))
    return np.stack(arrs, axis=0)


# ----------------------------------------------------------------------------
# 신호 / 추정기
# ----------------------------------------------------------------------------
def motion_signal(stack: np.ndarray) -> np.ndarray:
    """인접 프레임 절대차 평균. 길이 N-1, index i 는 전이 (i -> i+1).

    반환 d[i] 는 '프레임 i+1 에서 관측된 변화량' 으로 해석한다(뒤쪽 프레임에 귀속).
    """
    diff = np.abs(stack[1:] - stack[:-1])
    return diff.reshape(diff.shape[0], -1).mean(axis=1)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """길이 보존 이동평균 (edge 는 반사 패딩)."""
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="same")[pad:pad + len(x)]


# CCD/CrashBest 시간 prior: catalog/crashbest_videos.csv 1,500개 전수 실측 결과
#   first_crash_frame_index / frame_count 의 최소값 = 0.60 (min 30/50).
# 즉 충돌은 항상 클립 후반 ~40% 구간에서만 발생한다.
# 이 prior 로 초반 카메라 워밍/노이즈에서 나는 허위 피크를 배제한다.
CCD_MIN_FRAC = 0.55  # 0.60 에서 약간 여유를 둔 보수적 하한


def estimate_collision(nums: list[int], stack: np.ndarray, method: str, smooth_k: int,
                       min_frac: float = CCD_MIN_FRAC) -> int:
    """추정 collision_frame(원본 프레임 번호) 반환.

    method:
      argmax          : 전체 구간 원신호 argmax.
      smooth_argmax   : 이동평균 후 argmax (단발 노이즈 억제).
      windowed_argmax : CCD prior 구간([min_frac*N, end])으로 제한 후 평활 argmax.
    """
    d = motion_signal(stack)  # len N-1, index i -> frame nums[i+1]
    N = len(nums)
    if method == "argmax":
        peak_i = int(np.argmax(d))
    elif method == "smooth_argmax":
        peak_i = int(np.argmax(moving_average(d, smooth_k)))
    elif method == "windowed_argmax":
        ds = moving_average(d, smooth_k)
        start_frame = int(min_frac * N)          # 고려할 최소 프레임 idx
        valid_lo = max(0, start_frame - 1)       # 전이 idx i -> 프레임 i+1
        seg = ds[valid_lo:]
        peak_i = valid_lo + int(np.argmax(seg))
    else:
        raise ValueError(f"unknown method: {method}")
    frame_idx = min(peak_i + 1, N - 1)           # 변화가 관측된 뒤쪽 프레임
    return nums[frame_idx]


def baseline_midpoint(nums: list[int]) -> int:
    return nums[len(nums) // 2]


# ----------------------------------------------------------------------------
# 라벨
# ----------------------------------------------------------------------------
def load_labels(data_dir: Path) -> dict[str, int]:
    path = data_dir / "labels.csv"
    labels: dict[str, int] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["t_collision"])
            if t >= 0:
                labels[str(row["ID"])] = t
    return labels


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Stage2 충돌 프레임 국소화 로컬 측정")
    ap.add_argument("--data-dir", default=str(REPO / "data" / "stage2"))
    ap.add_argument("--fps", type=float, default=10.0, help="초 변환용 (CCD=10fps)")
    ap.add_argument("--size", type=int, default=96, help="다운스케일 정사각 크기")
    ap.add_argument("--smooth-k", type=int, default=3, help="smooth/windowed 이동평균 창")
    ap.add_argument("--min-frac", type=float, default=CCD_MIN_FRAC,
                    help="windowed_argmax 탐색 시작 비율 (CCD prior 근거 0.60)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    id_dirs = list_id_dirs(data_dir)
    labels = load_labels(data_dir)
    if not id_dirs:
        print(f"[stage2-eval] images 폴더 없음: {data_dir/'images'} — `make stage2-images` 필요")
        return 1
    if not labels:
        print("[stage2-eval] t_collision 라벨 없음 — 측정 불가")
        return 1

    methods = ["argmax", "smooth_argmax", "windowed_argmax"]
    errs = {m: [] for m in methods}
    errs["baseline_mid"] = []
    per_video = []

    for d in id_dirs:
        vid = d.name
        if vid not in labels:
            continue
        frames = list_frames(d)
        if len(frames) < 2:
            continue
        nums = [frame_number(f) for f in frames]
        stack = load_gray_stack(frames, args.size)
        gt = labels[vid]

        row = {"ID": vid, "gt": gt, "N": len(nums)}
        for m in methods:
            pred = estimate_collision(nums, stack, m, args.smooth_k, args.min_frac)
            errs[m].append(abs(pred - gt))
            row[m] = pred
        base = baseline_midpoint(nums)
        errs["baseline_mid"].append(abs(base - gt))
        row["baseline_mid"] = base
        per_video.append(row)

    # 표
    print(f"[stage2-eval] data_dir={data_dir}  videos={len(per_video)}  fps={args.fps}  size={args.size}")
    hdr = f"{'ID':>8} {'N':>3} {'gt':>4} {'argmax':>7} {'smooth':>7} {'window':>7} {'mid':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in per_video:
        print(f"{r['ID']:>8} {r['N']:>3} {r['gt']:>4} {r['argmax']:>7} "
              f"{r['smooth_argmax']:>7} {r['windowed_argmax']:>7} {r['baseline_mid']:>5}")

    def mae(key: str) -> float:
        v = errs[key]
        return float(np.mean(v)) if v else float("nan")

    print("\n-- 방법별 MAE --")
    for key in ["argmax", "smooth_argmax", "windowed_argmax", "baseline_mid"]:
        mf = mae(key)
        print(f"  {key:>15}: {mf:.3f} frames  ({mf/args.fps:.3f} s)")

    # 최선 방법 선택(프레임 MAE 최소) → 하니스 계약 METRIC
    best = min(["argmax", "smooth_argmax", "windowed_argmax"], key=lambda k: mae(k))
    best_mf = mae(best)
    base_mf = mae("baseline_mid")
    print(f"\n[stage2-eval] best_method={best}")
    print(f"METRIC collision_mae_frames={best_mf:.4f}")
    print(f"METRIC collision_mae_sec={best_mf/args.fps:.4f}")
    print(f"METRIC baseline_mid_mae_frames={base_mf:.4f}")
    print(f"METRIC baseline_mid_mae_sec={base_mf/args.fps:.4f}")
    improve = base_mf - best_mf
    print(f"METRIC improvement_frames_vs_baseline={improve:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
