"""Stage 1 추론 — predict_stage1(data_dir, model_dir) 계약.

근거: research/01-stage1-recapture/README.md, docs/03(제출 규격), baseline inference 인터페이스.
전략 요약: 잔차+시간 2-스트림 클립 분류. 오프라인이므로 weights=None + model/ 로컬 로드.
반환: pandas.DataFrame[ID, answer]  (answer in {ORIGINAL, RERECORDED})
"""
from __future__ import annotations

from pathlib import Path


def predict_stage1(data_dir, model_dir):
    """평가 서버가 호출하는 진입점. data_dir/videos/** 재귀, 파일 stem=ID.

    NOTE: 이것은 실행 가능한 스캐폴드다. 실제 모델 로드/추론은 학습 후 채운다.
    구조·제출 규격·경로 처리는 지금부터 올바르게 유지한다.
    """
    import pandas as pd

    from src.common import data as D
    from src.common.submit_guard import check_stage1

    videos = D.stage1_videos(data_dir)
    model_path = Path(model_dir) / "best.pt"

    rows = []
    if model_path.exists():
        # TODO(exp): 학습된 2-스트림 모델 로드 후 클립 단위 추론.
        #   model = build_model(weights=None); model.load_state_dict(torch.load(model_path)...)
        #   for v in videos: rows.append({"ID": v.stem, "answer": infer_one(model, v)})
        raise NotImplementedError("Stage1 모델 추론 미구현 — 학습 후 채울 것")
    else:
        # 모델 부재 시: 파이프라인 검증용 안전 기본값 (다수 클래스 가정).
        for v in videos:
            rows.append({"ID": v.stem, "answer": "ORIGINAL"})

    df = pd.DataFrame(rows, columns=["ID", "answer"])
    return check_stage1(df)


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage1"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage1"
    out = predict_stage1(dd, md)
    print(out.to_string(index=False))
    print(f"METRIC stage1_rows={len(out)}")
