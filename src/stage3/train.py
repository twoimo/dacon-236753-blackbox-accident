"""Stage 3 학습 스텁 — ego-motion 기반 가감속·조향 분류.

근거: research/03-stage3-egomotion/README.md, env/configs/stage3.yaml, docs/05.
계약: METRIC accel_acc=<v> + METRIC steer_acc_moving=<v> (STOPPED 제외 조향).

핵심 실험 순서:
  1) 20→10Hz 매핑 정확화 (여기서 틀리면 전부 무의미)
  2) 조향 부호(양수=좌회전) 검증 + 임계값 스윕
  3) 경량 백본(옵티컬플로우/X3D) 비교 후 앙상블
데이터: comma2k19 파생 라벨(data/stage3/labels_comma2k19.csv), 전체셋으로 확장.
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.runtime import resample_indices, set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage3 가감속·조향 학습")
    ap.add_argument("--config", default="stage3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    fr = cfg.get("framerate", {})
    print(f"[stage3] true_hz={fr.get('true_hz')} target_hz={fr.get('target_hz')}")

    if args.dry_run:
        # 프레임률 매핑 계약 검증 (docs/05 핵심 함정)
        src_hz = float(fr.get("true_hz", 20))
        tgt_hz = float(fr.get("target_hz", 10))
        idx = resample_indices(1200, src_hz, tgt_hz)
        assert len(idx) == int(1200 / src_hz * tgt_hz), "20→10Hz 리샘플 계약 불일치"
        assert cfg.get("labels", {}).get("steering_sign_check") is True, "조향 부호 검증 필수"
        print(f"[stage3] dry-run OK — {src_hz}Hz 1200f → {tgt_hz}Hz {len(idx)}샘플")
        print("METRIC dry_run_ok=1")
        return 0

    raise SystemExit("[stage3] 실제 학습 미구현. --dry-run 후 실험에서 구현.")


if __name__ == "__main__":
    raise SystemExit(main())
