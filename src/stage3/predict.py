"""Stage 3 추론 — predict_stage3(data_dir, model_dir) 계약.

근거: research/03-stage3-egomotion/README.md, docs/05(20→10Hz 함정), docs/03(제출).
입력: data_dir/videos/*.mp4 (파일 stem=ID).
반환: DataFrame[ID, sample_index, accel_label, steer_label]  (0.1초 단위).
핵심: 프레임 수 ffprobe 실측 → 10Hz 리샘플. STOPPED도 steer_label 필수.
"""
from __future__ import annotations

from pathlib import Path


def predict_stage3(data_dir, model_dir):
    import pandas as pd

    from src.common import data as D
    from src.common.runtime import count_frames, resample_indices
    from src.common.submit_guard import check_stage3

    videos = D.stage3_videos(data_dir)
    model_path = Path(model_dir) / "best.pt"

    rows = []
    for v in videos:
        vid = v.stem
        # docs/05: 컨테이너 fps 신뢰 금지 → 실측 후 20Hz 가정으로 10Hz 리샘플.
        try:
            n_src = count_frames(v)
        except Exception:
            n_src = 0
        idx = resample_indices(n_src, src_hz=20.0, target_hz=10.0) if n_src else [0]

        if model_path.exists():
            # TODO(exp): ego-motion 특징/옵티컬플로우 → accel(4)·steer(3) 분류.
            raise NotImplementedError("Stage3 모델 추론 미구현 — 학습 후 채울 것")

        for si, _f in enumerate(idx):
            rows.append({
                "ID": vid,
                "sample_index": si,
                "accel_label": "CONSTANT",
                "steer_label": "STRAIGHT",  # STOPPED 여부와 무관히 항상 채움
            })

    cols = ["ID", "sample_index", "accel_label", "steer_label"]
    df = pd.DataFrame(rows, columns=cols)
    return check_stage3(df)


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage3"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage3"
    out = predict_stage3(dd, md)
    print(out.head(10).to_string(index=False))
    print(f"METRIC stage3_rows={len(out)}")
