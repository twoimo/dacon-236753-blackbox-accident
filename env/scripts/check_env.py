#!/usr/bin/env python3
"""환경 자기진단 — 팀원이 온보딩 직후 1분 안에 기기 정합을 확인한다.

CUDA 유무와 무관하게 동작한다. 마지막에 하니스 계약대로 `METRIC env_ok=<0|1>` 을 출력한다.

사용:
    python env/scripts/check_env.py
근거: env/README.md, research/04-backbones-and-constraints/README.md
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 평가 서버 정합 대상 (docs/03 §6). 개발 기기에서 버전이 어긋나면 경고.
PINNED = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
}


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_python() -> bool:
    v = sys.version_info
    print(f"Python: {platform.python_version()} ({platform.machine()} / {platform.system()})")
    if v < (3, 11):
        warn("python 3.11+ 권장 (평가 서버 정합)")
        return True
    ok("python 버전 적합")
    return True


def check_pkg(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, "__version__", "?")


def check_packages() -> bool:
    print("패키지 버전 (평가 서버 핀 대조):")
    all_present = True
    for name, want in PINNED.items():
        got = check_pkg(name)
        if got is None:
            warn(f"{name} 미설치 (개발용이면 env/configs/requirements-train.txt 설치)")
            all_present = False
        elif got == want:
            ok(f"{name}=={got}")
        else:
            warn(f"{name}=={got} (서버 핀 {want} 과 다름)")
    return all_present


def check_device() -> str:
    torch = None
    try:
        torch = importlib.import_module("torch")
    except Exception:  # noqa: BLE001
        warn("torch 미설치 → 전처리/후처리 검증만 가능 (CUDA 학습 불가)")
        return "none"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        ok(f"CUDA 사용 가능: {name} (VRAM {vram:.1f} GiB)")
        if "L40S" not in name:
            warn("평가 서버는 L40S(44.7GiB). 기기가 다르면 VRAM/속도 프로파일 재확인 필요")
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        warn("MPS(Apple Silicon) 감지 → 전처리/후처리만. 학습·CUDA 추론은 GPU 기기에서")
        return "mps"
    warn("GPU 없음 → CPU 모드 (전처리/후처리 검증용)")
    return "cpu"


def check_tools() -> bool:
    print("외부 도구:")
    all_ok = True
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if path:
            ok(f"{tool}: {path}")
        else:
            warn(f"{tool} 없음 (Stage 1/3 영상 처리에 필요)")
            all_ok = False
    return all_ok


def check_data() -> bool:
    print("데이터/카탈로그:")
    catalog = REPO_ROOT / "catalog" / "catalog.json"
    if catalog.exists():
        ok(f"catalog.json 존재 ({catalog.stat().st_size} bytes)")
    else:
        fail("catalog/catalog.json 없음 — 저장소 구조 확인")
        return False
    ext = REPO_ROOT / "data" / "external"
    if ext.exists() and any(ext.iterdir()):
        ok("data/external 존재 (대용량 데이터 복원됨)")
    else:
        warn("data/external 비어있음 → 학습 시 `make data` 로 복원 (docs/07)")
    return True


def main() -> int:
    print("=" * 60)
    print("DACON 236753 환경 자기진단 (env/scripts/check_env.py)")
    print("=" * 60)
    results = [
        check_python(),
        check_packages(),
        check_tools(),
        check_data(),
    ]
    device = check_device()
    print("-" * 60)
    # env_ok: 저장소 구조 + python 이면 최소 통과 (GPU 없어도 리서치/전처리 가능)
    env_ok = 1 if (results[0] and results[3]) else 0
    print(f"판정: device={device}, 필수 통과={bool(env_ok)}")
    print(f"METRIC env_ok={env_ok}")
    print(f"ASI device={device}")
    return 0 if env_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
