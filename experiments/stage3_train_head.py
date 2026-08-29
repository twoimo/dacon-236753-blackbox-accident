#!/usr/bin/env python3
"""Stage 3 학습형 가감속·조향 분류기 (MPS GPU) — 무학습 베이스라인 초과 시도.

무학습(src/stage3/flow.py): accel_acc 0.642, steer_acc_moving 0.853 (majority 0.510/0.843).
여기서는 캐시된 흐름/외형 특징(experiments/cache/stage3_feats.npz)으로 시계열 분류기를 학습.

**중요 제약(정직)**: comma2k19 세그먼트가 1개(600샘플=60초)뿐이다. 랜덤 split 은 시간적으로
인접한 샘플이 train/test에 섞여 누설이 발생한다. 따라서 **시계열 분할(앞 70% 학습 / 뒤 30% 평가)**
과 **블록 교차검증(5 블록)** 두 방식으로 정직하게 평가한다.

모델: 프레임 특징(72) → 시간 윈도우(±W) 문맥 → MLP/1D-CNN → accel 4-class, steer 3-class 멀티태스크.
실행: .venv/bin/python -m experiments.stage3_train_head [--epochs 200]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments" / "cache" / "stage3_feats.npz"
MODEL_OUT = REPO / "model" / "stage3"
# 특징 추출 시 클래스 매핑 (stage3_extract_features.py)
ACCEL_NAMES = ["STOPPED", "DECELERATING", "CONSTANT", "ACCELERATING"]
STEER_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def windowed(X: np.ndarray, w: int) -> np.ndarray:
    """각 시점에 ±w 이웃 특징을 붙여 시간 문맥 부여 → (T, (2w+1)*F)."""
    T, F = X.shape
    pads = []
    for off in range(-w, w + 1):
        shifted = np.roll(X, -off, axis=0)
        if off < 0:
            shifted[:(-off)] = X[0]
        elif off > 0:
            shifted[-off:] = X[-1]
        pads.append(shifted)
    return np.concatenate(pads, axis=1)


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.LayerNorm(hid // 2), nn.Dropout(0.3),
        )
        self.accel = nn.Linear(hid // 2, 4)
        self.steer = nn.Linear(hid // 2, 3)

    def forward(self, x):
        h = self.trunk(x)
        return self.accel(h), self.steer(h)


def evaluate(model, Xte, a_te, s_te, dev):
    model.eval()
    with torch.no_grad():
        la, ls = model(torch.tensor(Xte, device=dev))
        pa = la.argmax(1).cpu().numpy(); ps = ls.argmax(1).cpu().numpy()
    accel_acc = float((pa == a_te).mean())
    # 조향은 STOPPED 제외 (대회 규약) — 여기 라벨엔 STOPPED 없음(전부 주행)
    moving = a_te != ACCEL_NAMES.index("STOPPED")
    steer_acc = float((ps[moving] == s_te[moving]).mean()) if moving.any() else float((ps == s_te).mean())
    return accel_acc, steer_acc, pa, ps


def train_eval(Xtr, atr, str_, Xte, ate, ste, dev, epochs, lr, seed, cw=True):
    torch.manual_seed(seed)
    model = MultiTaskHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # 클래스 가중 (steer 심한 불균형: STRAIGHT 84%)
    def weights(y, n):
        cnt = np.bincount(y, minlength=n).astype(np.float64)
        w = np.where(cnt > 0, cnt.sum() / (cnt + 1e-9), 0.0)
        w = w / w[w > 0].mean() if (w > 0).any() else w
        return torch.tensor(w, dtype=torch.float32, device=dev)
    wa = weights(atr, 4) if cw else None
    ws = weights(str_, 3) if cw else None
    Xt = torch.tensor(Xtr, device=dev)
    at = torch.tensor(atr, device=dev); st = torch.tensor(str_, device=dev)
    for ep in range(epochs):
        model.train()
        la, ls = model(Xt)
        loss = nn.functional.cross_entropy(la, at, weight=wa) + \
               nn.functional.cross_entropy(ls, st, weight=ws)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--window", type=int, default=5, help="시간 문맥 ±W 샘플")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(CACHE)
    X = d["X"].astype(np.float32); a = d["accel_y"]; s = d["steer_y"]
    T, F = X.shape
    # 정규화 (train 통계는 fold마다 재계산하나, 여기선 전체 표준화 후 fold 진행 — 특징이 저차원이라 영향 미미)
    Xw = windowed(X, args.window)
    dev = device()
    print(f"[s3-head] device={dev} T={T} F={F} → windowed {Xw.shape[1]}차원 (±{args.window})")
    print(f"[s3-head] accel dist={np.bincount(a, minlength=4)} steer dist={np.bincount(s, minlength=3)}")

    # 무학습 베이스라인 (RESULTS.md 실측치)
    BASE_ACCEL, BASE_STEER = 0.6417, 0.8533
    MAJ_ACCEL, MAJ_STEER = 0.5100, 0.8433

    # (1) 시계열 분할: 앞 70% train / 뒤 30% test (누설 없음)
    cut = int(0.7 * T)
    mu, sd = Xw[:cut].mean(0), Xw[:cut].std(0) + 1e-6
    Xn = (Xw - mu) / sd
    m = train_eval(Xn[:cut], a[:cut], s[:cut], Xn[cut:], a[cut:], s[cut:],
                   dev, args.epochs, args.lr, args.seed)
    aa, sa, _, _ = evaluate(m, Xn[cut:], a[cut:], s[cut:], dev)
    print(f"\n[s3-head] === 시계열 분할 (앞70%→뒤30%, n_test={T-cut}) ===")
    print(f"  learned : accel={aa:.4f}  steer_moving={sa:.4f}")
    print(f"  무학습  : accel={BASE_ACCEL:.4f}  steer={BASE_STEER:.4f}")
    print(f"  majority: accel={MAJ_ACCEL:.4f}  steer={MAJ_STEER:.4f}")

    # (2) 블록 교차검증 (5블록, 각 블록을 test로)
    nb = 5
    bounds = np.linspace(0, T, nb + 1).astype(int)
    accel_scores, steer_scores = [], []
    for b in range(nb):
        lo, hi = bounds[b], bounds[b + 1]
        te = np.zeros(T, bool); te[lo:hi] = True
        tr = ~te
        mu, sd = Xw[tr].mean(0), Xw[tr].std(0) + 1e-6
        Xn = (Xw - mu) / sd
        mb = train_eval(Xn[tr], a[tr], s[tr], Xn[te], a[te], s[te],
                        dev, args.epochs, args.lr, args.seed + b)
        aab, sab, _, _ = evaluate(mb, Xn[te], a[te], s[te], dev)
        accel_scores.append(aab); steer_scores.append(sab)
        print(f"  block{b+1}: accel={aab:.4f} steer={sab:.4f}")
    ma, ms = float(np.mean(accel_scores)), float(np.mean(steer_scores))
    print(f"\n[s3-head] === 블록 CV ({nb}블록) 평균 ===")
    print(f"  accel={ma:.4f} ± {np.std(accel_scores):.4f}  (무학습 {BASE_ACCEL:.4f})")
    print(f"  steer={ms:.4f} ± {np.std(steer_scores):.4f}  (무학습 {BASE_STEER:.4f})")
    print(f"METRIC stage3_head_accel_acc={ma:.4f}")
    print(f"METRIC stage3_head_steer_acc={ms:.4f}")
    print(f"METRIC stage3_head_mean_acc={(ma+ms)/2:.4f}")
    print(f"METRIC stage3_head_beats_flow={'1' if (ma+ms)/2 > (BASE_ACCEL+BASE_STEER)/2 else '0'}")

    # 최종 모델 저장 (전체 학습) — 개선된 경우만
    if (ma + ms) / 2 > (BASE_ACCEL + BASE_STEER) / 2:
        mu, sd = Xw.mean(0), Xw.std(0) + 1e-6
        final = train_eval((Xw - mu) / sd, a, s, (Xw - mu) / sd, a, s,
                          dev, args.epochs, args.lr, args.seed)
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": final.state_dict(), "in_dim": Xw.shape[1],
                    "window": args.window, "mu": mu, "sd": sd,
                    "accel_acc": ma, "steer_acc": ms}, MODEL_OUT / "behavior_head.pt")
        print(f"[s3-head] saved {MODEL_OUT/'behavior_head.pt'}")
    else:
        print("[s3-head] 무학습 베이스라인 초과 실패 → 모델 저장 안 함 (정직 기록)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
