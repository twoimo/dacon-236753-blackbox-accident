"""디바이스/시드/AMP 등 런타임 헬퍼.

근거: env/configs/common.yaml (runtime.*), research/04-backbones-and-constraints/README.md
로컬 macOS(CUDA 없음)에서도 import 가능하도록 torch 는 지연 import.
"""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """결정적 재현 (하니스 계약)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    """사용 가능한 최적 디바이스 반환. torch 없으면 문자열 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "auto") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device, enabled: bool = True):
    """AMP 컨텍스트. CUDA에서만 fp16, 그 외엔 no-op."""
    import contextlib

    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    dev_type = device.type if hasattr(device, "type") else str(device)
    if enabled and dev_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def count_frames(video_path: str | Path) -> int:
    """ffprobe -count_frames 로 실제 프레임 수 실측.

    docs/05: Stage3 컨테이너는 fps/duration 오선언 → 반드시 실측.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def resample_indices(n_src: int, src_hz: float, target_hz: float) -> list[int]:
    """src_hz 프레임 시퀀스를 target_hz 로 리샘플할 소스 프레임 인덱스.

    Stage3: 실제 20Hz → 대회 10Hz (docs/05). 시간축 2배 오차 방지의 핵심.
    """
    if src_hz <= 0 or target_hz <= 0:
        raise ValueError("hz는 양수여야 함")
    duration = n_src / src_hz
    n_target = int(duration * target_hz)
    step = src_hz / target_hz
    return [min(int(round(i * step)), n_src - 1) for i in range(n_target)]
