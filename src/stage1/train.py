"""Stage 1 학습 스텁 — 재촬영 합성 + 두 클래스 동일 재인코딩 + LODO 검증.

근거: research/01-stage1-recapture/README.md, env/configs/stage1.yaml
계약: METRIC lodo_macro_f1=<v>, METRIC macro_f1=<v> 출력.

이 파일은 실행 ��능한 골격이다. 실제 합성/학습 로직은 실험에서 채운다:
  1) CCD 프레임(data/external/CrashBest)에서 재촬영 합성 (물리>광학시뮬)
  2) 원본+재촬영을 동일 랜덤 ffmpeg 분포로 재인코딩 (코덱 누설 차단)
  3) 소스 영상 ID split + leave-one-device-out 홀드아웃
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.runtime import set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1 재녹화 판별 학습")
    ap.add_argument("--config", default="stage1")
    ap.add_argument("--dry-run", action="store_true", help="데이터/설정만 점검(모델 학습 없음)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    reencode = cfg.get("reencode", {})
    print(f"[stage1] config loaded: primary_metric={cfg.get('primary_metric')}")
    print(f"[stage1] reencode.apply_to_both_classes={reencode.get('apply_to_both_classes')}")

    if args.dry_run:
        # 설정 계약만 확인 (로컬 macOS에서도 통과)
        # 코덱 누설 차단: 원본+재촬영 동일 재인코딩 필수 (research/01)
        assert reencode.get("apply_to_both_classes") is True, "코덱 누설 차단(reencode.apply_to_both_classes) 설정 필수"
        print("[stage1] dry-run OK — 재촬영 합성/재인코딩 설정 확인됨")
        print("METRIC dry_run_ok=1")
        return 0

    # TODO(실험): 합성→재인코딩→학습→LODO 평가
    raise SystemExit("[stage1] 실제 학습 미구현. --dry-run 으로 설정 점검 후 실험에서 구현.")


if __name__ == "__main__":
    raise SystemExit(main())
