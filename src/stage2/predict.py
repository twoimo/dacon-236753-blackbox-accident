"""Stage 2 추론 — predict_stage2(data_dir, model_dir) 계약.

근거: research/02-stage2-anticipation/README.md, docs/03(제출 규격).
입력: data_dir/images/<ID>/frame_XXXXXX.jpg (Stage2만 이미지 폴더 입력, 재번호 금지).
반환: pandas.DataFrame[ID, collision_frame, entry_frame, evasion_space, entry_side]
전략: 충돌은 CCD 0/1 주석 기반 시간 국소화(라벨 확보), 진입/방향/회피는 약지도.
"""
from __future__ import annotations

from pathlib import Path


def predict_stage2(data_dir, model_dir):
    import pandas as pd

    from src.common import data as D
    from src.common.submit_guard import check_stage2

    id_dirs = D.stage2_image_dirs(data_dir)
    model_path = Path(model_dir) / "best.pt"

    rows = []
    frame_counts = {}
    for d in id_dirs:
        frames = D.stage2_frames(d)
        vid = d.name
        n = len(frames)
        # 원본 프레임 번호 범위 (재번호 금지: 파일명 숫자 사용)
        nums = [D.frame_number(f) for f in frames] or [0]
        frame_counts[vid] = n
        if model_path.exists():
            # TODO(exp): 시간 국소화 헤드로 충돌/진입 프레임 회귀, 방향/회피 분류.
            raise NotImplementedError("Stage2 모델 추론 미구현 — 학습 후 채울 것")
        # 스캐폴드 기본값: 중앙 프레임을 충돌로, 진입은 그 이전, 안전 기본 범주.
        mid = nums[len(nums) // 2]
        rows.append({
            "ID": vid,
            "collision_frame": mid,
            "entry_frame": nums[max(0, len(nums) // 2 - 5)],
            "evasion_space": 1,
            "entry_side": "RIGHT",
        })

    cols = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
    df = pd.DataFrame(rows, columns=cols)
    return check_stage2(df, frame_counts=frame_counts)


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage2"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage2"
    out = predict_stage2(dd, md)
    print(out.to_string(index=False))
    print(f"METRIC stage2_rows={len(out)}")
