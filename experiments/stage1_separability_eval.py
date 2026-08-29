#!/usr/bin/env python3
"""Stage 1 분리가능성 실측 — 코덱 중화 후 numpy 특징이 원본/재촬영을 구분하는가?

가중치 0.2 컴포넌트의 핵심 미검증 질문: 재촬영 예제가 진짜 재촬영본이 아니고(합성),
코덱 누설을 제거하면, 순수 numpy 특징만으로 두 클래스가 실제로 분리되는가?

방법 (모두 numpy/PIL, torch/cv2 불필요):
  1. CrashBest 프레임 N개를 소스로. 각 소스에서 ORIGINAL(원본)과 RERECAPTURE(합성) 쌍 생성.
  2. 코덱 누설 차단 위해 두 클래스를 동일 처리(재양자화)로 통일 — 픽셀 특징만 보게.
  3. 특징 추출: 잔차 고주파 에너지(라플라시안), FFT 고대역 비율, 엣지 밀도,
     블록 분산(모아레/노이즈 프록시).
  4. leave-one-source-out 대신, 소스 ID로 train/test 분리(누설 방지) 후
     간단한 로지스틱회귀(numpy 경사하강)로 Macro-F1 측정.

실행: .venv/bin/python -m experiments.stage1_separability_eval [--n 300]
출력: METRIC stage1_sep_macro_f1, per-feature 판별력(AUC 근사).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CCD = REPO / "data" / "external" / "CrashBest"

# 재촬영 합성기 재사용
sys.path.insert(0, str(REPO))
from experiments.stage1_synth_recapture import synthesize_recapture  # noqa: E402

IMG = 128  # 특징 추출용 해상도 (모아레/노이즈는 해상도 민감 → 너무 낮추지 않음)


def load_rgb(p: Path, size: int = IMG) -> np.ndarray:
    im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def requantize(img_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """코덱 누설 차단 모사: 두 클래스 모두 동일한 경미 JPEG류 재양자화 + 미세 블러.

    ORIGINAL 도 RERECAPTURE 도 같은 후처리를 거쳐 '재인코딩 여부'가 라벨과 무관하게 함.
    """
    from io import BytesIO

    # 동일 품질 JPEG 왕복 (양 클래스 공통) — 코덱/압축 단서 무력화
    q = int(rng.integers(70, 90))
    buf = BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _gray(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _laplacian_energy(g: np.ndarray) -> float:
    # 4-이웃 라플라시안 고주파 잔차 에너지
    lap = (
        -4 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    return float(np.mean(lap[1:-1, 1:-1] ** 2))


def _fft_highband_ratio(g: np.ndarray) -> float:
    # 고주파 대역 에너지 비율 (모아레/재양자화 흔적)
    F = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(F)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    high = mag[r > 0.5 * rmax].sum()
    total = mag.sum() + 1e-8
    return float(high / total)


def _edge_density(g: np.ndarray) -> float:
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float(gx + gy)


def _block_var(g: np.ndarray, b: int = 8) -> float:
    # 블록 내 분산의 평균 (재촬영 노이즈/격자 프록시)
    h, w = g.shape
    hh, ww = h // b * b, w // b * b
    gg = g[:hh, :ww].reshape(hh // b, b, ww // b, b)
    return float(gg.var(axis=(1, 3)).mean())


def features(img_u8: np.ndarray) -> np.ndarray:
    g = _gray(img_u8)
    return np.array([
        _laplacian_energy(g),
        _fft_highband_ratio(g),
        _edge_density(g),
        _block_var(g),
    ], dtype=np.float64)


FEAT_NAMES = ["lap_energy", "fft_high", "edge_density", "block_var"]


def standardize(X: np.ndarray, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def logreg_train(X, y, epochs=400, lr=0.1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = X.T @ (p - y) / n
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def macro_f1(y, pred):
    f1s = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def auc_approx(scores, y):
    # Mann-Whitney U 기반 AUC
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="소스 프레임 수 (각 원본+재촬영 쌍)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    srcs = sorted(CCD.glob("C_*.jpg"))
    if not srcs:
        print("CrashBest 프레임 없음"); return 1
    # 서로 다른 소스 영상에서 다양하게 추출 (파일명 C_<vid>_<idx>.jpg)
    rng.shuffle(srcs)
    srcs = srcs[: args.n]

    X, y, groups = [], [], []
    for i, p in enumerate(srcs):
        orig = load_rgb(p)
        recap = synthesize_recapture(orig, seed=int(rng.integers(1 << 30)))
        # 코덱 누설 차단: 두 클래스 동일 재양자화
        orig_q = requantize(orig, rng)
        recap_q = requantize(recap, rng)
        X.append(features(orig_q)); y.append(0); groups.append(p.stem)
        X.append(features(recap_q)); y.append(1); groups.append(p.stem)
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1} sources", flush=True)

    X = np.array(X); y = np.array(y)
    print(f"[stage1-sep] samples={len(y)} (orig={np.sum(y==0)}, recap={np.sum(y==1)})")

    # per-feature 판별력 (AUC)
    print("[stage1-sep] feature AUC (개별 판별력):")
    for j, name in enumerate(FEAT_NAMES):
        print(f"  {name:14s}: AUC={auc_approx(X[:, j], y):.3f}")

    # train/test 소스 분리 (앞 70% 소스 train) — 누설 방지
    n_src = args.n
    cut = int(0.7 * n_src)
    train_idx = [k for k in range(len(y)) if k // 2 < cut]
    test_idx = [k for k in range(len(y)) if k // 2 >= cut]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    Xtr_s, mu, sd = standardize(Xtr)
    Xte_s, _, _ = standardize(Xte, mu, sd)
    w, b = logreg_train(Xtr_s, ytr.astype(float))
    pred = (1 / (1 + np.exp(-(Xte_s @ w + b))) >= 0.5).astype(int)
    f1 = macro_f1(yte, pred)
    acc = float((pred == yte).mean())
    print(f"[stage1-sep] holdout: n_test={len(yte)} acc={acc:.3f}")
    print(f"METRIC stage1_sep_macro_f1={f1:.4f}")
    print(f"METRIC stage1_sep_acc={acc:.4f}")
    print(f"METRIC stage1_sep_baseline_f1=0.3333")  # 항상 한 클래스 시 macro-F1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
