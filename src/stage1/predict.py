"""Stage 1 추론 — predict_stage1(data_dir, model_dir) 계약.

근거: research/01-stage1-recapture/README.md, docs/03(제출 규격), baseline inference 인터페이스.
전략 요약 (2-스트림 잔차+시간, research §4):
  - Stream A (RGB/gray): 저주파 색/감마 시프트, 잔상(ghosting).
  - Stream B (제약 잔차): conv1 을 고정 Laplacian/고역통과(SRM류)로 초기화(yang2017laplacian)
    → 백본이 도로/차/날씨 같은 '콘텐츠'가 아니라 재촬영 지문을 보게 강제.
  - 시간 창 8~16 프레임(clip_len)으로 클립 단위 시공간 추론(X3D-S 등).
  - 입력에 컨테이너 메타데이터/비트레이트/파일크기 금지(코덱 누설 차단, docs/05 §1).
오프라인 규격: 가중치 인터넷 다운로드 불가 → weights=None + model_dir 로컬 로드.
반환: pandas.DataFrame[ID, answer]  (answer in {ORIGINAL, RERECORDED})
"""
from __future__ import annotations

from pathlib import Path


def predict_stage1(data_dir, model_dir):
    """평가 서버가 호출하는 진입점. data_dir/videos/** 재귀, 파일 stem=ID.

    모델 로드는 try/except 로 감싼다: 오프라인/의존성 부재(torch 미로드)/체크포인트 손상
    등 어떤 이유로 로드가 실패해도 파이프라인은 규격에 맞는 제출본을 반환해야 한다
    (제출 오류는 제출 횟수를 차감). 학습된 2-스트림 모델이 있으면 그것으로 추론하고,
    없거나 실패하면 안전 기본값으로 폴백한다.

    NOTE: 실행 가능한 스캐폴드다. 실제 2-스트림 로드/추론은 학습 후 _load_model/_infer_one
    에 채운다. 구조·제출 규격·경로 처리는 지금부터 올바르게 유지한다.
    """
    import pandas as pd

    from src.common import data as D
    from src.common.submit_guard import check_stage1

    videos = D.stage1_videos(data_dir)
    model_path = Path(model_dir) / "best.pt"

    model = None
    if model_path.exists():
        try:
            model = _load_model(model_path)
        except Exception as e:  # noqa: BLE001 - 어떤 실패든 안전 폴백
            print(f"[stage1] 모델 로드 실패 → 안전 기본값 폴백: {type(e).__name__}: {e}")
            model = None

    rows = []
    if model is not None:
        for v in videos:
            rows.append({"ID": v.stem, "answer": _infer_one(model, v)})
    else:
        # 모델 부재/로드 실패 시: 파이프라인 검증용 안전 기본값 (다수 클래스 가정).
        # '규격 통과용'이지 성능 제출용이 아니다 — 학습 후 model_dir 를 채운다.
        for v in videos:
            rows.append({"ID": v.stem, "answer": "ORIGINAL"})

    df = pd.DataFrame(rows, columns=["ID", "answer"])
    return check_stage1(df)


def _load_model(model_path: Path):
    """학습된 2-스트림 모델 로드. torch 필요(오프라인 GPU 평가 환경).

    로컬 macOS 개발기에서는 torch 가 로드되지 않을 수 있으므로 지연 import.
    실패 시 예외를 그대로 올려 predict_stage1 의 try/except 가 폴백하게 한다.
    """
    import torch  # 지연 import — 로컬에서 없으면 여기서 ImportError → 폴백

    _ = torch  # (스텁) 실제 로드 시 사용
    # TODO(exp): 학습 코드와 동일한 build_model(weights=None) 로 뼈대를 만든 뒤 로드.
    #   from src.stage1.model import build_two_stream
    #   model = build_two_stream(clip_len=16, residual_init="laplacian")
    #   state = torch.load(model_path, map_location="cpu")
    #   model.load_state_dict(state.get("model", state)); model.eval(); return model
    raise NotImplementedError(
        "Stage1 2-스트림 모델 로드 미구현 — src/stage1/train.py 로 학습 후 채운다."
    )


def _infer_one(model, video_path) -> str:
    """한 영상에서 클립(8~16프레임)을 샘플링해 재촬영 여부 추론.

    TODO(exp): ffmpeg 로 프레임 디코드 → clip_len 창 샘플 → 2-스트림 전방 →
    클립 로짓 평균/다수결 → {"ORIGINAL","RERECORDED"}.
    입력에 컨테이너 메타데이터를 절대 넣지 않는다(코덱 누설 차단).
    """
    raise NotImplementedError("Stage1 클립 추론 미구현")


if __name__ == "__main__":
    import sys

    dd = sys.argv[1] if len(sys.argv) > 1 else "data/stage1"
    md = sys.argv[2] if len(sys.argv) > 2 else "model/stage1"
    out = predict_stage1(dd, md)
    print(out.to_string(index=False))
    print(f"METRIC stage1_rows={len(out)}")
