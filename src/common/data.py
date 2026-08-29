"""데이터 경로/로딩 공통 유틸 — 평가 서버 레이아웃과 1:1.

근거: AGENTS.md(데이터 레이아웃), docs/02-data-spec.md
평가 시 data_dir 하위:
  Stage1: videos/**            (재귀, 파일 stem = ID)
  Stage2: images/<ID>/frame_XXXXXX.jpg
  Stage3: videos/*.mp4         (파일 stem = ID)
"""
from __future__ import annotations

from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".hevc"}


def stage1_videos(data_dir: str | Path) -> list[Path]:
    """Stage1: data_dir/videos 아래 모든 영상 (재귀)."""
    root = Path(data_dir) / "videos"
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def stage2_image_dirs(data_dir: str | Path) -> list[Path]:
    """Stage2: data_dir/images/<ID>/ 폴더 목록."""
    root = Path(data_dir) / "images"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def stage2_frames(id_dir: str | Path) -> list[Path]:
    """한 영상 폴더의 프레임 이미지 (파일명 순 = 원본 프레임 순)."""
    d = Path(id_dir)
    return sorted(p for p in d.glob("frame_*.jpg"))


def frame_number(frame_path: str | Path) -> int:
    """frame_000123.jpg -> 123 (원본 프레임 번호, 재번호 금지)."""
    stem = Path(frame_path).stem  # frame_000123
    return int(stem.split("_")[-1])


def stage3_videos(data_dir: str | Path) -> list[Path]:
    """Stage3: data_dir/videos/*.mp4 (파일 stem = ID)."""
    root = Path(data_dir) / "videos"
    return sorted(p for p in root.glob("*") if p.suffix.lower() in VIDEO_EXTS)


def video_id(path: str | Path) -> str:
    """파일 stem 을 ID로."""
    return Path(path).stem
