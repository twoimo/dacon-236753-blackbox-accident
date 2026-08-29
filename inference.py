"""제출용 통합 추론 진입점 — 평가 서버가 이 파일의 predict_stage{1,2,3} 를 호출한다.

근거: docs/03-evaluation-and-submission.md (필수 인터페이스), research/synthesis/.
규격:
  predict_stage1(data_dir, model_dir) -> DataFrame[ID, answer]
  predict_stage2(data_dir, model_dir) -> DataFrame[ID, collision_frame, entry_frame, evasion_space, entry_side]
  predict_stage3(data_dir, model_dir) -> DataFrame[ID, sample_index, accel_label, steer_label]

주의:
  - submit.zip 최상위에 이 파일 + model/ + requirements.txt 만.
  - 모든 모델은 weights=None 생성 후 model/ 의 .pt 로컬 로드 (인터넷 차단).
  - 이 파일은 src/ 를 import 한다. 패키징 시 src/ 도 zip 에 포함해야 한다
    (scripts/package_submit.sh 가 처리).
"""
from __future__ import annotations

import sys
from pathlib import Path

# 평가 서버에서 이 파일이 submit.zip 최상위에 풀리므로, 같은 위치의 src/ 를 경로에 추가.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage1.predict import predict_stage1  # noqa: E402
from src.stage2.predict import predict_stage2  # noqa: E402
from src.stage3.predict import predict_stage3  # noqa: E402

__all__ = ["predict_stage1", "predict_stage2", "predict_stage3"]
