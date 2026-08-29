"""Stage 3 옵티컬플로우 ego-motion 특징 — 학습 없이 numpy만으로.

근거: research/03-stage3-egomotion/README.md(경로 B: 플로우 통계→임계값),
      docs/05(20→10Hz 함정), env/configs/stage3.yaml(임계값 스윕).

설계 원칙 (로컬 측정 하니스와 GPU 추론이 동일 로직을 쓰도록 순수 numpy):
  - 프레임 추출: ffmpeg 로 그레이스케일 다운스케일 PNG (cv2 불필요).
  - 속도 프록시: Lucas-Kanade 스타일 전역 수직 흐름 |v_g| (전진 시 노면이
    화면 아래로 흐른다). 속도의 시간 미분 → 가감속.
  - 조향 프록시: 지평선 밴드에서 세로 스트립별 수평 LK 흐름의 중앙값 u_med.
    (양수 = 좌회전, comma2k19 CAN steering_angle 부호와 정합, corr≈+0.45.)

comma2k19 예제 세그먼트(600 샘플, 고속도로)에서 CAN 정답 대비 측정:
  accel_acc≈0.645(다수결 0.510), steer_acc≈0.850(다수결 0.843).
결정적: 모든 파라미터 고정, 랜덤 없음.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

# --- 고정 파라미터 (결정적) -------------------------------------------------
# 프레임 추출 해상도 (그레이스케일). 낮으면 흐름이 뭉개지고 높으면 느리다.
EXTRACT_W = 512
EXTRACT_H = 384
# 지평선 밴드 (조향용 수평 흐름) — 프레임 높이 비율.
HORIZON_Y0, HORIZON_Y1 = 0.25, 0.50
STRIP_W = 40  # 지평선 밴드 세로 스트립 폭(px, EXTRACT_W 기준)
# 스무딩/미분 파라미터 (속도는 매끄럽다는 물리 사전지식).
SPEED_SMOOTH = 25      # 속도 프록시 이동평균 창 (스윕 최적)
ACCEL_WIN = 24         # 가감속 중앙차분 반경(샘플) (스윕 최적)
STEER_SMOOTH = 9       # 조향 프록시 이동평균 창
# 조향 부호: +u_med = 좌회전 (CAN 정합). sign=+1.
STEER_SIGN = 1
# 기본 임계값 (z-score 기준; 스윕으로 갱신 가능).
DEFAULT_ACCEL_THR = 0.9
DEFAULT_STEER_THR_L = 2.2  # 스윕 최적
DEFAULT_STEER_THR_R = 2.0  # 스윕 최적


# --- 프레임 추출 ------------------------------------------------------------
def extract_frames_ffmpeg(video_path, out_dir, w=EXTRACT_W, h=EXTRACT_H):
    """ffmpeg 로 전 프레임을 그레이스케일 다운스케일 PNG 로 추출.

    docs/05 함정: 대회 mp4 는 fps 를 40 으로 오선언하고 후반 PTS 가 손상돼 있다.
      - fps 필터(fps=10)는 손상 PTS 때문에 프레임을 과다 드롭한다(600→480). 금지.
      - image2 머서 기본값도 비단조 DTS 를 만나면 프레임을 dedup 해 1200→~102 로 유실.
        → 반드시 `-fps_mode passthrough`(출력 옵션)로 디코드된 전 프레임을 그대로 쓴다.
        comma2k19 video.hevc 는 정상이라 어느 경우든 1200 이 나온다.
    전 프레임을 뽑고 호출부에서 stride-2 (20→10Hz) 서브샘플한다. f_%04d.png (1-based).
    반환: 정렬된 PNG 경로 리스트.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-fps_mode", "passthrough",   # 손상 PTS 프레임 유실 방지 (docs/05)
            "-vf", f"scale={w}:{h},format=gray",
            str(out_dir / "f_%04d.png"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("f_*.png"))


def load_gray(png_path):
    """PIL 로 그레이스케일 PNG → float64 2D 배열 (cv2 불필요)."""
    from PIL import Image

    return np.asarray(Image.open(png_path), dtype=np.float64)


# --- 저수준 흐름 연산 -------------------------------------------------------
def _blur3(a):
    """분리형 [1 2 1] 블러 — 그래디언트 안정화."""
    k = np.array([1.0, 2.0, 1.0])
    k /= k.sum()
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    return a


def _lk_region(a, b, y0, y1, x0, x1):
    """단일 영역 Lucas-Kanade 전역 흐름 (u=수평, v=수직) px 단위.

    최소제곱 정규방정식으로 한 창 안의 상수 흐름을 추정. det≈0 이면 0 반환.
    """
    A = a[y0:y1, x0:x1]
    B = b[y0:y1, x0:x1]
    Ix = np.zeros_like(A)
    Iy = np.zeros_like(A)
    Ix[:, 1:-1] = (A[:, 2:] - A[:, :-2]) * 0.5
    Iy[1:-1, :] = (A[2:, :] - A[:-2, :]) * 0.5
    It = B - A
    Sxx = (Ix * Ix).sum()
    Syy = (Iy * Iy).sum()
    Sxy = (Ix * Iy).sum()
    Sxt = (Ix * It).sum()
    Syt = (Iy * It).sum()
    det = Sxx * Syy - Sxy * Sxy
    if abs(det) < 1e-6:
        return 0.0, 0.0
    u = -(Syy * Sxt - Sxy * Syt) / det
    v = -(-Sxy * Sxt + Sxx * Syt) / det
    return float(u), float(v)


# --- 특징 추출 (샘플 시퀀스) -----------------------------------------------
def compute_flow_features(gray_frames):
    """10Hz 그레이스케일 프레임 시퀀스 → (speed_proxy, steer_proxy).

    gray_frames: (N,H,W) float 배열. 인접 샘플(0.1s) 간 흐름을 계산.
    반환:
      speed_proxy[N] = |전역 수직 LK 흐름|  (전진 속도 프록시)
      steer_proxy[N] = 지평선 스트립별 수평 LK 흐름의 중앙값 (+=좌회전)
    """
    frames = np.asarray(gray_frames, dtype=np.float64)
    n = len(frames)
    h, w = frames.shape[1:]
    blurred = np.stack([_blur3(f) for f in frames])

    v_g = np.zeros(n)
    u_med = np.zeros(n)
    hy0, hy1 = int(h * HORIZON_Y0), int(h * HORIZON_Y1)
    # STRIP_W 는 EXTRACT_W 기준 → 실제 폭에 비례 스케일.
    strip = max(8, int(STRIP_W * w / EXTRACT_W))
    for i in range(1, n):
        a, b = blurred[i - 1], blurred[i]
        _, v_g[i] = _lk_region(a, b, 0, h, 0, w)
        shifts = [
            _lk_region(frames[i - 1], frames[i], hy0, hy1, xs, xs + strip)[0]
            for xs in range(0, w - strip, strip)
        ]
        u_med[i] = float(np.median(shifts)) if shifts else 0.0
    return np.abs(v_g), u_med


def _smooth(x, k):
    if k <= 1:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="same")


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def classify_accel(speed_proxy, thr=DEFAULT_ACCEL_THR,
                   smooth=SPEED_SMOOTH, win=ACCEL_WIN):
    """속도 프록시 → 가감속 범주. 속도 스무딩 후 중앙차분 미분의 z-score.

    thr 초과 = ACCELERATING, -thr 미만 = DECELERATING, 그 외 CONSTANT.
    (STOPPED 는 영상 흐름만으로 신뢰 추정 불가 → 여기선 미판정. 예제
     세그먼트도 STOPPED 없음. predict 에서 별도 정지 판정 훅 가능.)
    """
    n = len(speed_proxy)
    sp = _smooth(speed_proxy, smooth)
    a = np.zeros(n)
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n - 1, i + win)
        a[i] = (sp[hi] - sp[lo]) / max(hi - lo, 1)
    az = _zscore(a)
    out = np.where(az > thr, "ACCELERATING",
                   np.where(az < -thr, "DECELERATING", "CONSTANT"))
    return out.tolist()


def classify_steer(steer_proxy, thr_l=DEFAULT_STEER_THR_L,
                   thr_r=DEFAULT_STEER_THR_R, smooth=STEER_SMOOTH,
                   sign=STEER_SIGN):
    """조향 프록시 → 조향 범주. 스무딩·부호 정렬 후 z-score 임계.

    +z > thr_l = LEFT, -z < -thr_r = RIGHT, 그 외 STRAIGHT.
    """
    z = _zscore(_smooth(np.asarray(steer_proxy) * sign, smooth))
    out = np.where(z > thr_l, "LEFT", np.where(z < -thr_r, "RIGHT", "STRAIGHT"))
    return out.tolist()
