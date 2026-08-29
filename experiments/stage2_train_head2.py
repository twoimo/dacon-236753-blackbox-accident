#!/usr/bin/env python3
"""Stage 2 학습형 충돌 국소화 v2 — 모션 신호 명시 결합 (MPS GPU).

v1(stage2_train_head.py)의 학습 헤드는 MAE 6.31f로 휴리스틱(5.22f)에 못 미쳤다.
원인: 32x32 flatten 특징이 공간구조 상실 + 모션(휴리스틱의 핵심 신호) 미활용.

v2 개선:
  - 입력에 모션 신호 d[t]와 그 1차차분(onset)을 명시 채널로 추가.
  - BiGRU로 양방향 시간 문맥.
  - 타깃: soft-gaussian + argmax(prior 창). 잔차 결합(학습 logit + 모션 onset prior).
  - video split, heuristic 동일 test set 비교.

실행: .venv/bin/python -m experiments.stage2_train_head2 [--epochs 60]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "experiments" / "cache" / "stage2_feats.npz"
MODEL_OUT = REPO / "model" / "stage2"
PRIOR = 0.60


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class MotionAwareHead(nn.Module):
    def __init__(self, feat_dim: int, emb: int = 96, gru: int = 96):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, emb), nn.ReLU(), nn.LayerNorm(emb))
        # 입력: 임베딩(emb) + 모션 d[t] + onset(Δd) = emb+2
        self.gru = nn.GRU(emb + 2, gru, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * gru, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, motion, onset):        # x:(B,T,F) motion,onset:(B,T)
        e = self.proj(x)
        z = torch.cat([e, motion.unsqueeze(-1), onset.unsqueeze(-1)], dim=-1)
        h, _ = self.gru(z)
        return self.head(h).squeeze(-1)          # (B,T)


def soft_gauss(y, T, sigma=2.0):
    idx = np.arange(T)[None, :]
    return np.exp(-0.5 * ((idx - y[:, None]) / sigma) ** 2).astype(np.float32)


def win_argmax(logits, n, prior):
    lo = int(prior * n)
    m = logits.copy(); m[:, :lo] = -1e9
    return m.argmax(axis=1)


def onset_pred(motion, T, prior=PRIOR):
    lo = int(prior * T)
    out = []
    for row in motion:
        seg = row[lo:]
        out.append(lo + int(np.argmax(np.diff(seg))) + 1 if len(seg) >= 2 else T // 2)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(CACHE, allow_pickle=True)
    X = d["X"].astype(np.float32); motion = d["motion"].astype(np.float32); y = d["y"].astype(np.int64)
    N, T, F = X.shape
    # onset 채널 = 모션 1차차분(양의 상승)
    onset = np.zeros_like(motion)
    onset[:, 1:] = np.clip(np.diff(motion, axis=1), 0, None)
    # 정규화
    motion_n = (motion - motion.mean()) / (motion.std() + 1e-6)
    onset_n = (onset - onset.mean()) / (onset.std() + 1e-6)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N); n_te = N // 5
    te, tr = perm[:n_te], perm[n_te:]
    dev = device(); print(f"[v2] device={dev} N={N} T={T} F={F}")

    def to(a, idx): return torch.tensor(a[idx], device=dev)
    Xtr, Xte = to(X, tr), to(X, te)
    Mtr, Mte = to(motion_n, tr), to(motion_n, te)
    Otr, Ote = to(onset_n, tr), to(onset_n, te)
    Ytr = torch.tensor(soft_gauss(y[tr], T, args.sigma), device=dev)
    ytr, yte = y[tr], y[te]

    model = MotionAwareHead(F).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    bs = 128; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); order = torch.randperm(len(tr)); tot = 0.0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            logit = model(Xtr[b], Mtr[b], Otr[b])
            loss = lossf(logit, Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"  ep{ep+1} loss={tot/len(tr):.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        lg = model(Xte, Mte, Ote).cpu().numpy()
    pred = win_argmax(lg, T, PRIOR)
    mae = float(np.abs(pred - yte).mean()); w3 = float((np.abs(pred - yte) <= 3).mean())
    h_pred = onset_pred(motion[te], T); h_mae = float(np.abs(h_pred - yte).mean())
    # 앙상블: 학습 확률의 prior창 argmax와 onset 평균
    ens = np.round(0.5 * pred + 0.5 * h_pred).astype(int)
    e_mae = float(np.abs(ens - yte).mean())

    print(f"\n[v2] === TEST n={len(yte)} (video split) ===")
    print(f"  heuristic onset : MAE={h_mae:.3f}f")
    print(f"  learned v2      : MAE={mae:.3f}f within3={w3:.3f}")
    print(f"  ensemble(50/50) : MAE={e_mae:.3f}f")
    best = min(mae, h_mae, e_mae)
    print(f"METRIC stage2_v2_mae_frames={mae:.4f}")
    print(f"METRIC stage2_v2_ens_mae_frames={e_mae:.4f}")
    print(f"METRIC stage2_v2_heuristic_mae={h_mae:.4f}")
    print(f"METRIC stage2_v2_best_mae={best:.4f}")
    print(f"METRIC stage2_v2_beats_heuristic={'1' if best < h_mae else '0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
