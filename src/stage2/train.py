"""Stage 2 학습 — 충돌 시간 국소화(지도) + 진입/방향/회피(약지도).

근거: research/02-stage2-anticipation/README.md, env/configs/stage2.yaml
계약: METRIC collision_mae_sec=<v> (낮을수록 좋음) + 보조 지표.

전략
----
  1) 충돌(collision_frame): CCD 첫 positive frame(catalog/crashbest_videos.csv 의
     first_crash_frame_index, 1,500개) 으로 지도학습 → 프레임별 충돌확률 시계열 회귀.
  2) 진입(entry_frame)/방향(entry_side): 객체추적(ultralytics)+차선 기하 약지도.
  3) 회피(evasion_space): 자유공간 heuristic (Stage3 depth 특징 공유 여지).

실측 근거 (experiments/stage2_collision_eval.py, 로컬 5샘플)
--------------------------------------------------------
학습된 헤드 없이도, 프레임간 모션 피크 + CCD 시간 prior(충돌은 항상 클립 후반 >=0.60N)
만으로 collision MAE 6.2→2.2 프레임(0.22s), naive midpoint(7.8) 대비 5.6프레임 개선.
아래 실제 학습 경로는 이 heuristic 을 상한이 아닌 하한(초기값)으로 두고 개선을 목표로 한다.

실제 학습 경로 (--dry-run 없이 실행 시)
---------------------------------------
데이터: CrashBest 프레임(catalog 로 인덱싱) + first_crash_frame_index soft-label.
  - 입력: 영상당 50프레임을 ResNet-18(weights=None, model/ 로컬 가중치) 로 프레임 특징화.
  - 시간모듈: BiGRU / temporal-conv 로 프레임별 충돌확률 로짓 산출.
  - 타깃: 충돌 프레임 주변 가우시안 soft-label (초 오차 채점과 정합; README §3.1).
  - 손실: soft-label BCE/KL + peak argmax 시각오차 정규화.
  - 추론 후처리: CCD 시간 prior 창 + argmax(= predict.estimate_attributes 와 동형).
GPU 필수(L40S). 로컬 macOS 는 torch 미가용(code-signing) → 아래에서 정보성 예외.
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.runtime import set_seed


def _real_train(cfg: dict) -> int:
    """실제 충돌 국소화 헤드 학습 (GPU 필요). torch 없으면 정보성 예외.

    현재 상태: 학습 파이프라인 뼈대만 문서화. 전체 CNN+BiGRU 학습은 평가서버급
    GPU(L40S)와 CrashBest 프레임 데이터(`make data`) 가 있어야 한다. 로컬 macOS 에서는
    torch 확장 로드가 code-signing 으로 막히므로 여기서 명확히 안내하고 종료한다.
    """
    try:
        import torch  # noqa: F401
    except Exception as e:  # ImportError 또는 code-signing dlopen 실패 모두 포함
        raise SystemExit(
            "[stage2] 실제 학습은 torch(GPU) 가 필요합니다. 현재 환경에서 torch 를 "
            f"로드할 수 없습니다 ({type(e).__name__}). 평가서버급 GPU 환경에서 "
            "`make data` 로 CrashBest 프레임을 복원한 뒤 실행하세요. "
            "로컬 검증은 experiments/stage2_collision_eval.py (numpy+PIL) 로 대체합니다."
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "[stage2] CUDA GPU 미탐지. 충돌 국소화 헤드(ResNet-18+BiGRU) 학습은 "
            "L40S 급 GPU 를 전제로 합니다. CPU 학습은 지원 범위 밖."
        )

    # --- 아래는 GPU 환경에서 채울 실제 학습 루프 자리 (스키마 확정 후 구현) ---
    #   1) CrashBest 프레임 로더 + first_crash_frame_index soft-label 타깃 생성
    #   2) ResNet-18(weights=None) 프레임 특징 → BiGRU 프레임별 충돌확률
    #   3) soft-label 손실 + 시각오차 정규화, 검증 MAE 로 best 선택
    #   4) model/stage2/best.pt + resnet18-f37072fd.pth 저장 (오프라인 로드용)
    raise SystemExit(
        "[stage2] 실제 학습 루프 미구현. 데이터/헤드 스키마 확정 후 이 경로를 채운다 "
        "(experiments/stage2_collision_eval.py 의 검증 지표를 목표로 개선)."
    )


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

    return _real_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
