"""Stage 1 실험 — Tier-2 광학 시뮬레이션 재촬영 합성기 (numpy/PIL).

근거: research/01-stage1-recapture/README.md §4 "재촬영 합성 파이프라인 (물리 우선)"
  Tier 2 (광학 시뮬): 서브픽셀 격자 + 감마 + 미세 호모그래피 + 렌즈 MTF(엣지 번짐)
  + Poisson/read 노이즈 + 리프레시-셔터 박자. **클립마다 격자 주파수·위상을 랜덤화.**

재촬영은 원본 위에 "디스플레이 → 광학계 → 센서" 라는 제2의 획득 체인이 덧씌워진 것.
이 합성기는 그 체인을 물리 순서대로 근사한다:

  1) 디스플레이 감마/EOTF     : 원본을 선형광으로 → 패널 감마 재적용 (색/감마 시프트)
  2) 서브픽셀 격자 (모아레원) : RGB 스트라이프/픽셀 그리드를 곱해 디스플레이 픽셀 구조 모사.
                                주파수·위상·방향을 호출마다 랜덤화 → 단일 주기 암기 방지.
  3) 미세 호모그래피          : 화면-카메라 비정렬(원근/회전/스케일) 약간.
  4) 렌즈 MTF (엣지 번짐)     : 가우시안 블러로 광학 저역통과 (모아레 안 보여도 남는 지문).
  5) 리프레시-셔터 박자       : 밝기 밴딩(수평 롤링) — 60/120Hz vs 셔터 위상.
  6) 센서 노이즈             : 신호 의존 Poisson(shot) + 가우시안 read 노이즈.
  7) 재양자화 (8-bit)         : 센서 ADC 근사.

크롭/재인코딩은 이 스크립트 밖에서 **두 클래스에 동일 적용**한다 (코덱 누설 차단;
stage1_codec_leak_check.py 참고). 여기서는 프레임 단위 픽셀 변환만 담당한다.

로컬 제약: cv2/torch 불가 → numpy + PIL 만 사용. 호모그래피는 PIL Image.transform 사용.
실행: .venv/bin/python -m experiments.stage1_synth_recapture [--n 2] [--src <dir>]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "data" / "external" / "CrashBest"


# --------------------------------------------------------------------------- #
# 개별 광학 효과 (모두 float32 [0,1] RGB HxWx3 로 동작)
# --------------------------------------------------------------------------- #
def _to_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def _to_u8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)


def display_gamma(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """디스플레이 EOTF 근사: 선형광 왕복 + 약한 감마/대비 시프트."""
    # 원본을 대략 선형광으로 (sRGB 감마 역), 패널 감마를 살짝 다르게 재적용
    lin = np.power(np.clip(img, 1e-6, 1.0), 2.2)
    panel_gamma = rng.uniform(2.0, 2.6)
    out = np.power(np.clip(lin, 1e-6, 1.0), 1.0 / panel_gamma)
    # 패널 밝기/대비 미세 변동
    gain = rng.uniform(0.92, 1.08)
    bias = rng.uniform(-0.02, 0.02)
    return np.clip(out * gain + bias, 0.0, 1.0)


def subpixel_grid(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """디스플레이 서브픽셀/픽셀 격자 곱셈 변조. 모아레의 물리적 원천.

    주파수·위상·방향을 랜덤화해 단일 합성 주기 암기를 막는다 (research/01 §3, luo2021sadg).
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 화면 픽셀 피치 (원본 픽셀당 몇 사이클): 재촬영 스케일에 따라 달라짐
    freq = rng.uniform(0.20, 0.48)          # 사이클/픽셀 (나이퀴스트 근처에서 모아레 강함)
    theta = rng.uniform(0.0, np.pi)         # 격자 방향
    phase = rng.uniform(0.0, 2 * np.pi)
    depth = rng.uniform(0.03, 0.12)         # 변조 깊이
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    grid = 1.0 + depth * np.sin(2 * np.pi * freq * proj + phase)
    out = img * grid[..., None]
    # RGB 스트라이프 위상차 (서브픽셀): 채널별로 아주 살짝 다른 위상
    for c in range(3):
        cphase = phase + (c - 1) * rng.uniform(0.3, 0.9)
        cgrid = 1.0 + 0.4 * depth * np.sin(2 * np.pi * freq * proj + cphase)
        out[..., c] *= cgrid
    return np.clip(out, 0.0, 1.0)


def micro_homography(pil: Image.Image, rng: np.random.Generator) -> Image.Image:
    """화면-카메라 비정렬: 약한 원근/회전/스케일 (PIL 퍼스펙티브 변환)."""
    w, h = pil.size
    # 코너를 소폭 흔든다 (최대 변 길이의 ~1.5%)
    jitter = 0.015
    def jp(x, y):
        return (x + rng.uniform(-jitter, jitter) * w,
                y + rng.uniform(-jitter, jitter) * h)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [jp(*p) for p in src]
    coeffs = _perspective_coeffs(dst, src)
    return pil.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)


def _perspective_coeffs(src, dst):
    """PIL PERSPECTIVE 계수 (dst->src 매핑) 계산."""
    matrix = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for p in src for c in p], dtype=np.float64)
    res = np.linalg.solve(A, B)
    return res.tolist()


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def lens_mtf_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """렌즈+디스플레이 MTF 저역통과: 분리 가우시안 블러 (엣지 스프레드 변화).

    모아레가 안 보이는 '깨끗한' 재촬영도 엣지 지문은 남는다
    (thongkamwitoon2015recapture). numpy 분리 합성곱으로 구현.
    """
    sigma = rng.uniform(0.6, 1.4)
    k = _gaussian_kernel1d(sigma)
    out = img
    # 가로 방향
    out = _convolve_axis(out, k, axis=1)
    # 세로 방향
    out = _convolve_axis(out, k, axis=0)
    return np.clip(out, 0.0, 1.0)


def _convolve_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """1D 커널을 지정 축으로 분리 합성곱 (edge padding)."""
    r = len(k) // 2
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        sl = [slice(None)] * 3
        sl[axis] = slice(i, i + img.shape[axis])
        out += w * p[tuple(sl)]
    return out


def refresh_beat(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """리프레시율 vs 셔터 박자: 수평 밝기 밴딩(롤링). 시간 아티팩트의 프레임내 흔적."""
    h = img.shape[0]
    y = np.arange(h, dtype=np.float32)
    band_freq = rng.uniform(1.5, 6.0) / h    # 화면 높이당 밴드 수
    phase = rng.uniform(0.0, 2 * np.pi)
    amp = rng.uniform(0.01, 0.05)
    band = 1.0 + amp * np.sin(2 * np.pi * band_freq * y * h + phase)
    return np.clip(img * band[:, None, None], 0.0, 1.0)


def sensor_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """신호 의존 Poisson(shot) + 가우시안 read 노이즈 (센서 획득)."""
    # shot noise: 광자수 스케일 후 포아송
    peak = rng.uniform(120.0, 600.0)          # 밝은 화소의 등가 광자수
    photons = np.clip(img, 0.0, 1.0) * peak
    shot = rng.poisson(photons).astype(np.float32) / peak
    # read noise: 가법 가우시안
    read_sigma = rng.uniform(0.002, 0.012)
    out = shot + rng.normal(0.0, read_sigma, size=img.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 파이프라인
# --------------------------------------------------------------------------- #
def synthesize_recapture(src_u8: np.ndarray, seed: int | None = None) -> np.ndarray:
    """CrashBest 프레임(HxWx3 uint8)을 재촬영본(uint8)으로 변환. 호출마다 랜덤화.

    물리 순서: 디스플레이(감마→서브픽셀 격자) → 광학(호모그래피→MTF 블러)
             → 시간(리프레시 박자) → 센서(노이즈) → 재양자화(8bit).
    """
    rng = np.random.default_rng(seed)
    img = _to_float(src_u8)

    # 1) 디스플레이 감마/EOTF
    img = display_gamma(img, rng)
    # 2) 서브픽셀 격자 (모아레 원천)
    img = subpixel_grid(img, rng)
    # 3) 미세 호모그래피 (PIL 경유)
    pil = Image.fromarray(_to_u8(img), mode="RGB")
    pil = micro_homography(pil, rng)
    img = _to_float(np.asarray(pil))
    # 4) 렌즈/디스플레이 MTF 저역통과
    img = lens_mtf_blur(img, rng)
    # 5) 리프레시-셔터 박자 밴딩
    img = refresh_beat(img, rng)
    # 6) 센서 노이즈
    img = sensor_noise(img, rng)
    # 7) 8-bit 재양자화 (ADC)
    return _to_u8(img)


def _pick_sources(src_dir: Path, n: int) -> list[Path]:
    frames = sorted(src_dir.glob("C_*.jpg"))
    if not frames:
        return []
    # 서로 다른 소스 영상에서 뽑아 다양성 확보 (파일명 C_<vid>_<idx>.jpg)
    step = max(1, len(frames) // max(1, n))
    return [frames[min(i * step, len(frames) - 1)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1 Tier-2 광학 재촬영 합성기")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="CrashBest 프레임 디렉터리")
    ap.add_argument("--n", type=int, default=2, help="합성할 샘플 수")
    ap.add_argument("--out", default=None, help="출력 디렉터리 (기본: temp)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir = Path(args.src)
    sources = _pick_sources(src_dir, args.n)
    if not sources:
        print(f"[synth] CrashBest 프레임을 찾지 못함: {src_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="stage1_synth_recapture_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[synth] 소스: {src_dir}  샘플: {len(sources)}  출력: {out_dir}")
    made: list[tuple[str, str]] = []
    for i, sp in enumerate(sources):
        src_u8 = np.asarray(Image.open(sp).convert("RGB"))
        rec_u8 = synthesize_recapture(src_u8, seed=args.seed + i)
        before_path = out_dir / f"{sp.stem}_before.png"
        after_path = out_dir / f"{sp.stem}_after_recapture.png"
        Image.fromarray(src_u8, "RGB").save(before_path)
        Image.fromarray(rec_u8, "RGB").save(after_path)
        # 변화량 지표 (동일성 검증): 평균 절대차
        mad = float(np.abs(src_u8.astype(np.int16) - rec_u8.astype(np.int16)).mean())
        made.append((str(before_path), str(after_path)))
        print(f"[synth] {sp.name}: before={before_path.name} after={after_path.name} "
              f"MAD={mad:.2f} (0이 아니면 실제 변환됨)")

    print("[synth] 샘플 경로:")
    for b, a in made:
        print(f"    BEFORE {b}")
        print(f"    AFTER  {a}")
    print(f"METRIC synth_samples={len(made)}")
    print(f"METRIC synth_out_dir_ok={1 if out_dir.exists() else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
