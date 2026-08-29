"""Stage 3 — 학습 없는 optical-flow ego-motion 베이스라인 측정 하니스.

근거: research/03-stage3-egomotion/README.md, env/configs/stage3.yaml, docs/05.

목표: comma2k19 예제 세그먼트 영상만으로 0.1초 단위 가감속·조향 범주를 추정하고,
CAN 파생 라벨(data/stage3/labels_comma2k19.csv, 600행)에 대해 실측 정확도를 측정한다.

환경 제약(로컬 macOS 검증됨): numpy + PIL + ffmpeg 만 사용. cv2/torch/pandas/scipy/av 미탑재.
  - ffmpeg 로 그레이스케일 프레임 추출 (컨테이너 fps 오선언 → -count_frames 로 실측)
  - 20Hz 실측 → 10Hz 리샘플은 프레임 stride=2 (frame_index = 2 * sample_index).
    docs/05: fps 필터(fps=10)는 손상된 PTS 때문에 600이 아닌 480 프레임을 뽑으므로 금지.
  - optical flow 는 순수 numpy Lucas-Kanade — src/stage3/flow.py 공용 로직 재사용.

특징 (CAN 지상진실 대비 상관 검증):
  - accel: 전역 수직 LK 플로우 |v_g| → 속도 프록시 → 시간 미분 → 가감속 (corr(speed) ~0.31)
  - steer: 지평선 밴드에서 열-스트립별 수평 LK 이동량의 중앙값 u_med (corr(steer) ~0.45)

출력 계약(하니스): METRIC accel_acc=, METRIC steer_acc_moving=, METRIC mean_acc=,
  그리고 스윕에서 찾은 최적 임계값.

실행:  .venv/bin/python -m experiments.stage3_flow_eval
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.stage3 import flow as F  # 공용 흐름 로직 (predict.py 와 동일)

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / ("data/external/comma2k19/Example_1/"
                "b0c9d2329ad1606b|2018-08-02--08-34-47/40/video.hevc")
LABELS = REPO / "data/stage3/labels_comma2k19.csv"


# --- 프레임 로드 (ffmpeg 추출 → 20→10Hz stride-2) ---------------------------
def count_frames(video: Path) -> int:
    """docs/05: 컨테이너 fps/duration 신뢰 금지 → -count_frames 실측."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def load_10hz(video: Path, out_dir: Path) -> np.ndarray:
    """20Hz 전체 추출 → 10Hz 샘플(stride=2). shape (N,H,W) float64.

    라벨 frame_index = 2 * sample_index 규약과 정합.
    """
    n_src = count_frames(video)
    files = F.extract_frames_ffmpeg(video, out_dir)
    if len(files) != n_src:
        files = files[:n_src]  # 디코드 수 != 실측이면 실측 기준 절단
    sel = files[0::2]  # 10Hz
    return np.stack([np.asarray(Image.open(f), dtype=np.float64) for f in sel])


# --- 신호 → 범주 (스윕용; flow.py 와 동일 공식) -----------------------------
def _smooth(x, k):
    return np.asarray(x, float) if k <= 1 else np.convolve(x, np.ones(k) / k, mode="same")


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def predict_accel(speed, sm, win, thr):
    sp = _smooth(speed, sm)
    n = len(sp)
    d = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - win), min(n - 1, i + win)
        d[i] = (sp[hi] - sp[lo]) / max(hi - lo, 1)
    d = _z(d)
    return ["ACCELERATING" if d[i] > thr else ("DECELERATING" if d[i] < -thr else "CONSTANT")
            for i in range(n)]


def predict_steer(u_med, sm, thr_l, thr_r, sign):
    ue = _z(_smooth(np.asarray(u_med) * sign, sm))
    return ["LEFT" if ue[i] > thr_l else ("RIGHT" if ue[i] < -thr_r else "STRAIGHT")
            for i in range(len(ue))]


# --- 평가 -------------------------------------------------------------------
def load_labels(path: Path):
    accel, steer = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            accel.append(row["accel_label"])
            steer.append(row["steer_label"])
    return accel, steer


def accuracy(pred, gt, mask=None):
    idx = range(len(gt)) if mask is None else [i for i in range(len(gt)) if mask[i]]
    if not idx:
        return 0.0
    return sum(pred[i] == gt[i] for i in idx) / len(idx)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage3flow_") as td:
        imgs = load_10hz(VIDEO, Path(td))
    speed, u_med = F.compute_flow_features(imgs)
    n = len(speed)

    accel_gt, steer_gt = load_labels(LABELS)
    assert len(accel_gt) == n, f"라벨 {len(accel_gt)} != 샘플 {n}"

    # 조향은 STOPPED 제외 채점(docs/03). 이 세그먼트는 STOPPED 없음 → 전체 moving.
    moving = [a != "STOPPED" for a in accel_gt]
    n_move = sum(moving)

    maj_accel = max(set(accel_gt), key=accel_gt.count)
    maj_steer = max(set(steer_gt), key=steer_gt.count)
    base_accel = accel_gt.count(maj_accel) / n
    base_steer = sum(steer_gt[i] == maj_steer for i in range(n) if moving[i]) / n_move

    # --- 임계값 스윕 (env/configs/stage3.yaml labels.sweep 정신) ---
    # 물리 임계값(mps2/deg)은 CAN 라벨 생성용. 영상 프록시는 z-score 임계값을 스윕.
    best_a = (-1.0, None)
    for sm in (17, 21, 25, 29):
        for win in (14, 16, 20, 24):
            for thr in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
                acc = accuracy(predict_accel(speed, sm, win, thr), accel_gt)
                if acc > best_a[0]:
                    best_a = (acc, (sm, win, thr))

    best_s = (-1.0, None)
    steer_thr = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)
    for sm in (5, 9, 13, 17):
        for sign in (1, -1):
            for tl in steer_thr:
                for tr in steer_thr:
                    acc = accuracy(predict_steer(u_med, sm, tl, tr, sign), steer_gt, moving)
                    if acc > best_s[0]:
                        best_s = (acc, (sm, sign, tl, tr))

    accel_acc, (a_sm, a_win, a_thr) = best_a
    steer_acc_moving, (s_sm, s_sign, s_tl, s_tr) = best_s
    mean_acc = (accel_acc + steer_acc_moving) / 2.0

    print(f"[stage3-flow] samples={n} (STOPPED={accel_gt.count('STOPPED')})")
    print(f"[stage3-flow] accel majority={maj_accel} base={base_accel:.4f}")
    print(f"[stage3-flow] steer majority={maj_steer} base(moving)={base_steer:.4f}")
    print(f"[stage3-flow] BEST accel cfg: smooth={a_sm} win={a_win} zthr={a_thr}")
    print(f"[stage3-flow] BEST steer cfg: smooth={s_sm} sign={s_sign:+d} "
          f"zthr_L={s_tl} zthr_R={s_tr}")
    print(f"METRIC accel_acc={accel_acc:.4f}")
    print(f"METRIC steer_acc_moving={steer_acc_moving:.4f}")
    print(f"METRIC mean_acc={mean_acc:.4f}")
    print(f"METRIC accel_gain_vs_majority={accel_acc - base_accel:+.4f}")
    print(f"METRIC steer_gain_vs_majority={steer_acc_moving - base_steer:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
