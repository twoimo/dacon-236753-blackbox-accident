#!/usr/bin/env python3
"""Stage 2 학습형 충돌 헤드 — 5-fold 교차검증 + 최종 모델 저장 (MPS GPU).

stage2_train_head2 의 motion-aware BiGRU 를 5-fold video split 로 검증해 견고성을 확인하고,
전체 데이터로 최종 모델을 학습해 model/stage2/collision_head.pt 로 저장한다.
추론은 학습 logit(prior 창 argmax)과 motion-onset 을 50/50 앙상블.

실행: .venv/bin/python -m experiments.stage2_cv_train [--epochs 60]
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
GRID = 32


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class MotionAwareHead(nn.Module):
    def __init__(self, feat_dim: int, emb: int = 96, gru: int = 96):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, emb), nn.ReLU(), nn.LayerNorm(emb))
        self.gru = nn.GRU(emb + 2, gru, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * gru, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, motion, onset):
        e = self.proj(x)
        z = torch.cat([e, motion.unsqueeze(-1), onset.unsqueeze(-1)], dim=-1)
        h, _ = self.gru(z)
        return self.head(h).squeeze(-1)


def soft_gauss(y, T, sigma):
    idx = np.arange(T)[None, :]
    return np.exp(-0.5 * ((idx - y[:, None]) / sigma) ** 2).astype(np.float32)


def win_argmax(logits, T, prior):
    lo = int(prior * T); m = logits.copy(); m[:, :lo] = -1e9
    return m.argmax(axis=1)


def onset_pred(motion, T, prior=PRIOR):
    lo = int(prior * T); out = []
    for row in motion:
        seg = row[lo:]
        out.append(lo + int(np.argmax(np.diff(seg))) + 1 if len(seg) >= 2 else T // 2)
    return np.array(out)


def prep(d):
    X = d["X"].astype(np.float32); motion = d["motion"].astype(np.float32); y = d["y"].astype(np.int64)
    onset = np.zeros_like(motion); onset[:, 1:] = np.clip(np.diff(motion, axis=1), 0, None)
    mn = (motion - motion.mean()) / (motion.std() + 1e-6)
    on = (onset - onset.mean()) / (onset.std() + 1e-6)
    return X, motion, mn, on, y


def train_one(Xtr, Mtr, Otr, Ytr, F, dev, epochs, lr, seed):
    torch.manual_seed(seed)
    model = MotionAwareHead(F).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    bs = 128
    for ep in range(epochs):
        model.train(); order = torch.randperm(Xtr.shape[0])
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            loss = lossf(model(Xtr[b], Mtr[b], Otr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    d = np.load(CACHE, allow_pickle=True)
    X, motion, mn, on, y = prep(d)
    N, T, F = X.shape
    dev = device()
    print(f"[cv] device={dev} N={N} T={T} F={F} folds={args.folds}")

    Xt = torch.tensor(X, device=dev)
    Mt = torch.tensor(mn, device=dev)
    Ot = torch.tensor(on, device=dev)
    Yt = torch.tensor(soft_gauss(y, T, args.sigma), device=dev)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    folds = np.array_split(perm, args.folds)

    head_maes, ens_maes, heur_maes, w3s = [], [], [], []
    t0 = time.time()
    for fi in range(args.folds):
        te = folds[fi]; tr = np.concatenate([folds[j] for j in range(args.folds) if j != fi])
        tr_t = torch.tensor(tr, device=dev)
        model = train_one(Xt[tr_t], Mt[tr_t], Ot[tr_t], Yt[tr_t], F, dev, args.epochs, args.lr, args.seed + fi)
        model.eval()
        with torch.no_grad():
            lg = model(Xt[torch.tensor(te, device=dev)], Mt[torch.tensor(te, device=dev)],
                       Ot[torch.tensor(te, device=dev)]).cpu().numpy()
        pred = win_argmax(lg, T, PRIOR)
        h = onset_pred(motion[te], T)
        ens = np.round(0.5 * pred + 0.5 * h).astype(int)
        yte = y[te]
        head_maes.append(np.abs(pred - yte).mean())
        heur_maes.append(np.abs(h - yte).mean())
        ens_maes.append(np.abs(ens - yte).mean())
        w3s.append((np.abs(ens - yte) <= 3).mean())
        print(f"  fold{fi+1}: head={head_maes[-1]:.3f} heur={heur_maes[-1]:.3f} ens={ens_maes[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[cv] === {args.folds}-fold 평균 ===")
    print(f"  heuristic : {np.mean(heur_maes):.3f} ± {np.std(heur_maes):.3f} f")
    print(f"  head      : {np.mean(head_maes):.3f} ± {np.std(head_maes):.3f} f")
    print(f"  ensemble  : {np.mean(ens_maes):.3f} ± {np.std(ens_maes):.3f} f  within3={np.mean(w3s):.3f}")
    print(f"METRIC stage2_cv_ensemble_mae_frames={np.mean(ens_maes):.4f}")
    print(f"METRIC stage2_cv_ensemble_mae_sec={np.mean(ens_maes)/10:.4f}")
    print(f"METRIC stage2_cv_heuristic_mae_frames={np.mean(heur_maes):.4f}")
    print(f"METRIC stage2_cv_within3={np.mean(w3s):.4f}")
    print(f"METRIC stage2_cv_beats_heuristic={'1' if np.mean(ens_maes) < np.mean(heur_maes) else '0'}")

    # 최종: 전체 데이터로 학습해 저장
    print("[cv] 전체 데이터로 최종 모델 학습 → 저장")
    final = train_one(Xt, Mt, Ot, Yt, F, dev, args.epochs, args.lr, args.seed)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final.state_dict(), "feat_dim": F, "grid": GRID,
        "prior": PRIOR, "ensemble_weight": 0.5, "sigma": args.sigma,
        "cv_ensemble_mae_frames": float(np.mean(ens_maes)),
        "arch": "MotionAwareHead(emb96,gru96,bidir)",
    }, MODEL_OUT / "collision_head.pt")
    print(f"[cv] saved {MODEL_OUT/'collision_head.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
