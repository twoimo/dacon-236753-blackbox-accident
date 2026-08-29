"""실험 CLI 엔트리포인트 — 학습/추론을 하니스 계약(METRIC)으로 감싼다.

근거: env/README.md(하니스 계약), research/synthesis/README.md.
사용:
    python -m experiments.run train  --stage stage1 [--dry-run]
    python -m experiments.run predict --stage stage3 --data-dir data/stage3 --model-dir model/stage3
"""
from __future__ import annotations

import argparse
import importlib
import time


def _run_train(stage: str, dry_run: bool) -> int:
    mod = importlib.import_module(f"src.{stage}.train")
    import sys

    argv = ["train", "--config", stage]
    if dry_run:
        argv.append("--dry-run")
    sys.argv = argv
    return mod.main()


def _run_predict(stage: str, data_dir: str, model_dir: str) -> int:
    mod = importlib.import_module(f"src.{stage}.predict")
    fn = getattr(mod, f"predict_{stage}")
    t0 = time.time()
    df = fn(data_dir, model_dir)
    elapsed = time.time() - t0
    print(df.head(5).to_string(index=False))
    print(f"METRIC {stage}_rows={len(df)}")
    print(f"METRIC {stage}_predict_sec={elapsed:.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DACON 236753 실험 러너")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="Stage 학습")
    p_tr.add_argument("--stage", required=True, choices=["stage1", "stage2", "stage3"])
    p_tr.add_argument("--dry-run", action="store_true")

    p_pr = sub.add_parser("predict", help="Stage 추론")
    p_pr.add_argument("--stage", required=True, choices=["stage1", "stage2", "stage3"])
    p_pr.add_argument("--data-dir", required=True)
    p_pr.add_argument("--model-dir", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        return _run_train(args.stage, args.dry_run)
    if args.cmd == "predict":
        return _run_predict(args.stage, args.data_dir, args.model_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
