"""Stage 2 추론 — predict_stage2(data_dir, model_dir) 계약.

근거: research/02-stage2-anticipation/README.md, docs/03(제출 규격), env/configs/stage2.yaml.
입력: data_dir/images/<ID>/frame_XXXXXX.jpg (Stage2만 이미지 폴더 입력, 재번호 금지).
반환: pandas.DataFrame[ID, collision_frame, entry_frame, evasion_space, entry_side]

전략 (experiments/stage2_collision_eval.py 로 로컬 실측)
------------------------------------------------------
- collision_frame: 프레임 간 전역 변화(global motion energy)의 피크로 시간 국소화.
    * 인접 프레임 그레이스케일 절대차 평균 d[t] 의 argmax.
    * CCD 시간 prior 로 창 제한: catalog/crashbest_videos.csv 1,500개 전수에서
      first_crash_frame_index/frame_count 최소=0.60 → 충돌은 항상 클립 후반.
      초반 카메라 워밍/노이즈 허위 피크를 배제(로컬 5샘플 MAE 6.2→2.2 프레임).
    * 프레임 번호는 파일명 숫자 그대로(재번호 금지). 로컬 0-기반이라 label 과 동일 좌표계.
- entry_frame: CCD 정답 라벨 없음 → 약지도. 문서화된 heuristic 으로 collision 이전
    고정 오프셋(ENTRY_OFFSET) 프레임을 사용하고 [nums[0], collision] 로 clamp.
- entry_side: CCD 정답 없음 → 약지도. 충돌 직전 창의 좌/우 절반 모션 비대칭으로 추정.
    검증 라벨이 없으므로(labels.csv 전부 -1) 실패 시 안전 기본값(RIGHT).
- evasion_space: 가장 불확실 → 안전 기본값(0=없음). 보고서용 heuristic 확장 여지.

의존성 가드
----------
- 로컬 macOS .venv 는 code-signing 으로 pandas/cv2/torch 로드가 막힌다.
  본 함수는 numpy+PIL 만으로 collision/entry/side 를 계산하고, DataFrame 조립·검증만
  pandas/submit_guard 로 한다. torch 헤드(model_dir/best.pt)가 있으면 우선 사용하되
  현재 미구현 → heuristic 폴백(제출 오류로 횟수 차감되지 않도록 예외를 던지지 않음).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# CCD/CrashBest 시간 prior (catalog/crashbest_videos.csv 1,500개 실측): 충돌은 클립 후반.
# 1,500영상 전수 그리드서치(experiments/stage2_ccd_large_eval.py) 결과:
#   prior 0.60 + onset(모션 상승엣) 방식이 MAE 5.22f(0.52s)로 최적
#   (window argmax 5.58f, midpoint baseline 12.19f 대비 개선), within-3f 0.48.
CCD_MIN_FRAC = 0.60        # first_crash/frame_count min=0.60 — 실측 최적
DOWNSCALE = 96             # 그레이스케일 다운스케일 정사각 크기
SMOOTH_K = 3               # 모션 신호 이동평균 창
USE_ONSET = True           # True=모션 상승엣(충돌 시작) / False=피크 argmax
ENTRY_OFFSET = 10          # entry_frame = collision - 오프셋 (약지도, 라벨 없음)
DEFAULT_EVASION = 0        # 안전 기본값 (0=없음)
DEFAULT_SIDE = "RIGHT"     # 안전 기본값 (허용 범주)


# ----------------------------------------------------------------------------
# numpy/PIL 수치 코어 (pandas 불필요 — 로컬에서 그대로 테스트 가능)
# ----------------------------------------------------------------------------
def _load_gray_stack(frame_paths, size: int = DOWNSCALE) -> np.ndarray:
    """프레임 경로 리스트 → (N, size, size) float32 그레이스케일 스택."""
    from PIL import Image

    arrs = []
    for p in frame_paths:
        im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
        arrs.append(np.asarray(im, dtype=np.float32))
    return np.stack(arrs, axis=0)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="same")[pad:pad + len(x)]


def _motion_signal(stack: np.ndarray) -> np.ndarray:
    """인접 프레임 절대차 평균. 길이 N-1, index i = 전이 (i -> i+1)."""
    diff = np.abs(stack[1:] - stack[:-1])
    return diff.reshape(diff.shape[0], -1).mean(axis=1)


def _windowed_peak_index(d: np.ndarray, n_frames: int, min_frac: float, smooth_k: int) -> int:
    """CCD prior 창([min_frac*N, end])에서 충돌 전이 index 반환.

    USE_ONSET=True: 평활 신호의 1차차분(상승 기울기) 최대 지점 = 충돌 시작.
      1,500영상 실측에서 argmax(피크)보다 MAE 낮음(5.22 vs 5.58f).
    USE_ONSET=False: 종래 피크 argmax.
    """
    ds = _moving_average(d, smooth_k)
    start_frame = int(min_frac * n_frames)
    valid_lo = max(0, min(start_frame - 1, len(ds) - 1))
    seg = ds[valid_lo:]
    if seg.size == 0:
        return valid_lo
    if USE_ONSET and seg.size >= 2:
        # 모션이 급증하는 지점(상승 기울기 최대) = 충돌 순간
        return valid_lo + int(np.argmax(np.diff(seg))) + 1
    return valid_lo + int(np.argmax(seg))


def estimate_attributes(nums, stack,
                        min_frac: float = CCD_MIN_FRAC,
                        smooth_k: int = SMOOTH_K,
                        entry_offset: int = ENTRY_OFFSET):
    """(collision_frame, entry_frame, evasion_space, entry_side) 산출 — 원본 프레임 번호.

    experiments/stage2_collision_eval.py 의 windowed_argmax 와 동일 로직.
    """
    n = len(nums)
    if n == 1:
        c = nums[0]
        return c, max(0, c), DEFAULT_EVASION, DEFAULT_SIDE

    d = _motion_signal(stack)                       # len n-1
    peak_i = _windowed_peak_index(d, n, min_frac, smooth_k)
    coll_idx = min(peak_i + 1, n - 1)               # 변화가 관측된 뒤쪽 프레임 idx
    collision_frame = nums[coll_idx]

    # entry_frame: 약지도(라벨 없음) — collision 이전 고정 오프셋.
    entry_idx = max(0, coll_idx - entry_offset)
    entry_frame = nums[entry_idx]

    # entry_side: 충돌 직전 창의 좌/우 모션 비대칭 (약지도, 검증 라벨 없음).
    side = DEFAULT_SIDE
    try:
        w0 = max(0, coll_idx - 8)
        win = np.abs(stack[w0 + 1:coll_idx + 1] - stack[w0:coll_idx])  # (m,H,W)
        if win.size:
            half = win.shape[2] // 2
            left = float(win[:, :, :half].mean())
            right = float(win[:, :, half:].mean())
            side = "LEFT" if left > right else "RIGHT"
    except Exception:
        side = DEFAULT_SIDE

    return collision_frame, entry_frame, DEFAULT_EVASION, side


# ----------------------------------------------------------------------------
# 제출 계약 (pandas + submit_guard)
# ----------------------------------------------------------------------------
def predict_stage2(data_dir, model_dir):
    import pandas as pd

    from src.common import data as D
    from src.common.submit_guard import check_stage2

    id_dirs = D.stage2_image_dirs(data_dir)
    model_path = Path(model_dir) / "best.pt"

    # torch 헤드 우선 (있으면). 현재 미구현 → heuristic 폴백.
    torch_head = None
    if model_path.exists():
        try:
            torch_head = _try_load_torch_head(model_path)
        except Exception:
            torch_head = None

    rows = []
    frame_counts = {}
    for d in id_dirs:
        vid = d.name
        frames = D.stage2_frames(d)
        nums = [D.frame_number(f) for f in frames] or [0]
        frame_counts[vid] = len(frames)

        if not frames:
            # 프레임 없음 — 결측 오답 방지 위해 안전 기본값으로 행 채움.
            rows.append({"ID": vid, "collision_frame": 0, "entry_frame": 0,
                         "evasion_space": DEFAULT_EVASION, "entry_side": DEFAULT_SIDE})
            continue

        stack = _load_gray_stack(frames)
        if torch_head is not None:
            coll, entry, evas, side = torch_head(stack, nums)
        else:
            coll, entry, evas, side = estimate_attributes(nums, stack)

        rows.append({
            "ID": vid,
            "collision_frame": coll,
            "entry_frame": entry,
            "evasion_space": evas,
            "entry_side": side,
        })

    cols = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
    df = pd.DataFrame(rows, columns=cols)
    # 후처리 방어: 범위 clamp / 정수화 / 범주 화이트리스트 (docs/03 채점 함정).
    return check_stage2(df, frame_counts=frame_counts)


def _try_load_torch_head(model_path: Path):
    """학습된 시간 국소화 헤드 로더 (미구현 자리표시).

    반환 시 callable(stack, nums) -> (coll, entry, evas, side).
    현재는 헤드 스키마 미확정 → NotImplementedError (호출부에서 heuristic 폴백).
    """
    raise NotImplementedError("Stage2 torch 헤드 미구현 — heuristic 폴백 사용")


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage2"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage2"
    out = predict_stage2(dd, md)
    print(out.to_string(index=False))
    print(f"METRIC stage2_rows={len(out)}")
