"""실험 설정 로더 — env/configs/*.yaml 을 읽어 공통(common) 위에 Stage 설정을 병합한다.

근거: env/configs/, research/synthesis/README.md
사용:
    from src.common.config import load_config
    cfg = load_config("stage1")   # common.yaml + stage1.yaml 병합
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyyaml 필요: pip install pyyaml") from e

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "env" / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """override 를 base 위에 재귀 병합 (override 우선)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stage: str) -> dict[str, Any]:
    """common.yaml + stage{N}.yaml 병합 결과 반환.

    stage: "stage1" | "stage2" | "stage3" (또는 "common")
    """
    common_path = CONFIG_DIR / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8")) or {}
    if stage == "common":
        return common
    stage_path = CONFIG_DIR / f"{stage}.yaml"
    if not stage_path.exists():
        raise FileNotFoundError(f"설정 없음: {stage_path}")
    stage_cfg = yaml.safe_load(stage_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(common, stage_cfg)
    merged["_stage"] = stage
    return merged


if __name__ == "__main__":
    import json
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    print(json.dumps(load_config(stage), ensure_ascii=False, indent=2))
