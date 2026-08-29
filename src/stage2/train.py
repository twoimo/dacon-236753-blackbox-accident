"""Stage 2 학습 스텁 — 충돌 시간 국소화 + 진입/방향/회피 약지도.

근거: research/02-stage2-anticipation/README.md, env/configs/stage2.yaml
계약: METRIC collision_mae_sec=<v> (낮을수록 좋음) + 보조 지표.

전략:
  1) 충돌: CCD 첫 positive frame(catalog/crashbest_videos.csv)로 지도 → 시간 국소화.
  2) 진입/방향: 추적(ultralytics)+차선 기하 약지도.
  3) 회피: 자유공간 heuristic (Stage3 depth 특징 공유 여지).
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.runtime import set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage2 사고 시점·상황 학습")
    ap.add_argument("--config", default="stage2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    print(f"[stage2] primary_metric={cfg.get('primary_metric')}")
    print(f"[stage2] collision_head.backbone={cfg.get('collision_head', {}).get('backbone')}")

    if args.dry_run:
        # 설정 계약 확인: 충돌 국소화 백본 + 후처리 방어
        assert cfg.get("collision_head", {}).get("backbone"), "collision_head.backbone 미설정"
        assert cfg.get("postprocess", {}).get("integerize") is True, "정수화 후처리 필수"
        print("[stage2] dry-run OK — 충돌 국소화/약지도 설정 확인됨")
        print("METRIC dry_run_ok=1")
        return 0

    raise SystemExit("[stage2] 실제 학습 미구현. --dry-run 후 실험에서 구현.")


if __name__ == "__main__":
    raise SystemExit(main())
