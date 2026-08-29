"""Stage 1 학습 스텁 — 재촬영 합성 + 두 클래스 동일 재인코딩 + LODO 검증.

근거: research/01-stage1-recapture/README.md, env/configs/stage1.yaml
계약: METRIC lodo_macro_f1=<v>, METRIC macro_f1=<v> 출력.

이 파일은 실행 가능한 골격이다. 실제 합성/학습 로직은 실험에서 채운다:
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

    return _train_real(cfg)


def _train_real(cfg) -> int:
    """실제 학습 진입점 — 합성→재인코딩→LODO. GPU 학습은 torch 가용 시에만.

    계획(research/01 §4, 검증된 실험 훅크는 experiments/ 에 이미 구현):
      1) 합성: CCD 75k 프레임(data/external/CrashBest)에서 Tier-2 광학 시뮬 재촬영
         생성 — experiments/stage1_synth_recapture.synth_recapture (클립마다 파라미터 랜덤).
      2) 재인코딩: 원본+재촬영을 동일 ffmpeg 분포로 통과해 코덱 누설 제거
         — experiments/stage1_codec_leak_check 가 before=1→after=0 로 입증함.
      3) 검증: source_video_id 기준 split + 최소 1화면/1카메라 홀드아웃(LODO).
         그 홀드아웃 Macro-F1 이 진짜 지표 — METRIC lodo_macro_f1=<v>.
      4) 모델: 2-스트림(잔차+시간). conv1 고정 Laplacian 초기화, clip_len 8~16.

    torch 가 없는 로컬(code-signing) 환경에서는 안내 메시지와 함께 비제로 종료한다.
    """
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[stage1] torch 미가동({type(e).__name__}) — GPU 학습 불가.")
        print("[stage1] 합성/재인코딩 검증은 로컬에서 가능:")
        print("         .venv/bin/python -m experiments.stage1_codec_leak_check")
        print("         .venv/bin/python -m experiments.stage1_synth_recapture")
        print("[stage1] GPU 환경(torch)에서 다시 실행하면 학습이 가동된다.")
        raise SystemExit("[stage1] torch 부재로 실학습 중단 (로컬 macOS 예상 동작).")

    # torch 가용 환경(GPU 서버)에서만 도달.
    # TODO(exp): synth_recapture 로 포지티브 생성 → 두 클래스 동일 재인코딩 →
    #   source_video_id split + LODO 홀드아웃 → 2-스트림 학습 →
    #   print(f"METRIC macro_f1={{...}}"); print(f"METRIC lodo_macro_f1={{...}}")
    raise SystemExit("[stage1] 2-스트림 학습 루프 미구현 — 합성/재인코딩 훅크는 experiments/ 에 검증됨.")


if __name__ == "__main__":
    raise SystemExit(main())
