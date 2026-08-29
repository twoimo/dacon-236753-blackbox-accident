#!/usr/bin/env python3
"""Stage 2 학습형 충돌 국소화 헤드 (MPS GPU) — 휴리스틱(MAE 5.22f) 초과 목표.

캐시된 프레임 특징(experiments/cache/stage2_feats.npz, stage2_extract_features.py 산출)으로
프레임별 충돌확률 시계열을 학습한다. soft-gaussian 타깃(충돌 프레임 주변 부드러운 확률).
video 기준 train/val split(프레임 누설 방지, research/01·02 프로토콜).

모델: per-frame 특징(1024) → 임베딩 → 1D temporal conv → 프레임별 충돌 logit.
추론: argmax(확률) = 충돌 프레임. MAE(프레임/초) 를 휴리스틱과 비교.

사용: .venv/bin/python -m experiments.stage2_train_head [--epochs 40]
출력: METRIC stage2_head_mae_frames / mae_sec, val within-3f, 휴리스틱 대비 개선.
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
FPS = 10.0


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def soft_gaussian_target(gt: int, length: int, sigma: float = 2.0) -> np.ndarray:
    idx = np.arange(length, dtype=np.float32)
    t = np.exp(-0.5 * ((idx - gt) / sigma) ** 2)
    t[length:] = 0.0
    s = t.sum()
    return t / s if s > 0 else t


class CollisionHead(nn.Module):
    def __init__(self, f_in=1024, emb=128, hidden=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(f_in, emb), nn.ReLU(), nn.LayerNorm(emb),
        )
        # 프레임 특징 + 모션 스칼라 → temporal conv
        self.tconv = nn.Sequential(
            nn.Conv1d(emb + 1, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, motion):
        # x: (B,T,F), motion: (B,T)
        e = self.embed(x)                       # (B,T,emb)
        z = torch.cat([e, motion.unsqueeze(-1)], dim=-1)  # (B,T,emb+1)
        z = z.transpose(1, 2)                   # (B,emb+1,T)
        logit = self.tconv(z).squeeze(1)        # (B,T)
        return logit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--prior", type=float, default=0.55, help="추론 시 CCD 시간 prior 하한")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(CACHE, allow_pickle=True)
    X, motion, lengths, y = d["X"], d["motion"], d["lengths"], d["y"]
    valid = lengths > 0
    X, motion, lengths, y = X[valid], motion[valid], lengths[valid], y[valid]
    n, T, F = X.shape
    dev = get_device()
    print(f"[train] device={dev} videos={n} T={T} F={F}")

    # video split 80/20 (프레임 누설 없음: 애초에 video 단위 표본)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]

    # soft-gaussian 타깃
    Y = np.zeros((n, T), dtype=np.float32)
    mask = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        L = int(lengths[i])
        Y[i] = soft_gaussian_target(int(y[i]), T, args.sigma)
        mask[i, :L] = 1.0

    Xt = torch.tensor(X, device=dev)
    Mt = torch.tensor(motion, device=dev)
    Yt = torch.tensor(Y, device=dev)
    Kt = torch.tensor(mask, device=dev)

    model = CollisionHead(f_in=F).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def masked_ce(logit, target, m):
        # 프레임 마스크 적용 log-softmax 크로스엔트로피(soft target)
        logit = logit.masked_fill(m < 0.5, -1e9)
        logp = torch.log_softmax(logit, dim=1)
        return -(target * logp * m).sum() / m.sum()

    tr_t = torch.tensor(tr, device=dev)
    va_t = torch.tensor(va, device=dev)
    bs = 128
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr)
        tr_t = torch.tensor(tr, device=dev)
        losses = []
        for b in range(0, len(tr), bs):
            idx = tr_t[b:b + bs]
            logit = model(Xt[idx], Mt[idx])
            loss = masked_ce(logit, Yt[idx], Kt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs} loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 평가 (val) — argmax + prior 창
    model.eval()
    with torch.no_grad():
        logit = model(Xt[va_t], Mt[va_t]).cpu().numpy()
    errs, errs_prior = [], []
    for j, i in enumerate(va):
        L = int(lengths[i]); gt = int(y[i])
        lg = logit[j][:L].copy()
        pred = int(np.argmax(lg))
        errs.append(abs(pred - gt))
        lo = int(args.prior * L)
        lg2 = lg.copy(); lg2[:lo] = -1e9
        errs_prior.append(abs((lo + int(np.argmax(lg2[lo:]))) - gt))
    errs = np.array(errs); errs_prior = np.array(errs_prior)
    best = min(errs.mean(), errs_prior.mean())
    use_prior = errs_prior.mean() < errs.mean()

    print(f"\n[train] === val ({len(va)} videos) ===")
    print(f"  argmax       MAE={errs.mean():.3f}f  within3={(errs<=3).mean():.3f}")
    print(f"  argmax+prior MAE={errs_prior.mean():.3f}f  within3={(errs_prior<=3).mean():.3f}")
    print(f"  휴리스틱 참고: onset+prior0.60 = 5.22f (전체 1500)")
    print(f"METRIC stage2_head_mae_frames={best:.4f}")
    print(f"METRIC stage2_head_mae_sec={best/FPS:.4f}")
    print(f"METRIC stage2_head_within3={max((errs<=3).mean(),(errs_prior<=3).mean()):.4f}")
    print(f"METRIC stage2_head_use_prior={int(use_prior)}")

    # 모델 저장 (predict 가 로드)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "f_in": F,
                "use_prior": use_prior, "prior": args.prior,
                "val_mae_frames": float(best)}, MODEL_OUT / "best.pt")
    print(f"[train] saved {MODEL_OUT/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
