#!/usr/bin/env python3
"""AI/LLM 이 색인할 수 있는 형태로 전체 데이터셋 카탈로그를 생성한다.

생성물 (모두 catalog/ 아래, 전부 결정론적)
  catalog.json          단일 정본. 대회 메타 + 데이터셋 + 스테이지 스키마 + 요약 통계
  files.csv             CrashBest 이미지를 제외한 모든 파일의 평면 인덱스(+sha256)
  media_index.csv       모든 영상/이미지의 컨테이너·코덱·해상도·프레임수
  crashbest_index.csv   CrashBest 75,000장 × (프레임 라벨 + 영상 메타 조인)
  crashbest_videos.csv  CrashBest 1,500 영상 단위 집계
  integrity.json        중복/결측/불일치 판정 결과

사용
  python scripts/build_catalog.py                 # 전체 생성
  python scripts/build_catalog.py --skip-hash     # sha256 생략(빠른 갱신)
  python scripts/build_catalog.py --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

CATALOG_VERSION = "1.0.0"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".hevc", ".3gp", ".3gpp", ".wmv"}
# 카탈로그에서 제외할 경로.
# VCS·환경·캐시와 함께 배포/학습 산출물(dist, model, output …)도 제외한다.
# 이들은 데이터셋이 아니라 데이터셋에서 파생된 것이므로 색인 대상이 아니다.
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".DS_Store", "node_modules",
    ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "dist", "model", "output", "runs", "wandb", "checkpoints",
    ".gjc", ".kiro", ".claude",
}
CRASHBEST_REL = "data/external/CrashBest"


# --------------------------------------------------------------------------- utils
def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name == ".DS_Store":
                continue
            yield Path(dirpath) / name


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def classify(rel: str, suffix: str) -> tuple[str, str]:
    """(dataset, category) 결정."""
    parts = rel.split("/")
    if parts[0] == "data" and len(parts) > 1:
        if parts[1] == "external":
            dataset = parts[2] if len(parts) > 2 else "external"
            dataset = "CCD" if dataset in {"CrashBest", "Crash_Table.csv"} else dataset
        elif parts[1].startswith("stage"):
            dataset = f"competition_{parts[1]}"
        else:
            # data/ 바로 아래의 파일(예: data/SOURCES.md)
            dataset = "competition_meta"
    elif parts[0] == "baseline":
        dataset = "baseline"
    elif parts[0] in {"docs", "catalog", "scripts"}:
        dataset = parts[0]
    else:
        dataset = "root"

    if suffix in VIDEO_EXT:
        category = "video"
    elif suffix in IMAGE_EXT:
        category = "image"
    elif suffix == ".csv":
        category = "labels" if rel.endswith("labels.csv") or "labels" in Path(rel).stem else "table"
    elif suffix in {".md", ".txt"}:
        category = "document"
    elif suffix == ".ipynb":
        category = "notebook"
    elif suffix in {".py", ".sh"}:
        category = "code"
    elif suffix == ".json":
        category = "index"
    else:
        category = "other"
    return dataset, category


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt"
        ":format=format_name,duration,bit_rate",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return {}
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "pix_fmt": stream.get("pix_fmt"),
        "container": fmt.get("format_name"),
        "duration_s": fmt.get("duration"),
        "bit_rate": fmt.get("bit_rate"),
    }


def count_frames(path: Path) -> int | None:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        return int(result.stdout.strip())
    except Exception:
        return None


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


# --------------------------------------------------------------------------- CrashBest
def build_crashbest(root: Path, out_dir: Path, jobs: int, do_hash: bool) -> dict:
    image_dir = root / CRASHBEST_REL
    table = root / "data" / "external" / "Crash_Table.csv"
    if not image_dir.is_dir():
        return {"present": False}

    annotations: dict[str, dict[str, str]] = {}
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as handle:
            annotations = {row["vidname"]: row for row in csv.DictReader(handle)}

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    print(f"  CrashBest: {len(images):,}장 스캔 (hash={'on' if do_hash else 'off'}, jobs={jobs})")

    def describe(path: Path) -> dict:
        stem = path.stem                       # C_000001_01
        _, _, remainder = stem.partition("_")  # 000001_01
        video_id, _, frame_text = remainder.rpartition("_")
        frame_no = int(frame_text)             # 1-기반
        annotation = annotations.get(video_id, {})
        width, height = image_size(path)
        crash_flag = annotation.get(f"frame_{frame_no}")
        return {
            "path": f"{CRASHBEST_REL}/{path.name}",
            "video_id": video_id,
            "frame_no": frame_no,
            "frame_index": frame_no - 1,
            "is_crash_frame": int(crash_flag) if crash_flag not in (None, "") else "",
            "timing": annotation.get("timing", ""),
            "weather": annotation.get("weather", ""),
            "egoinvolve": annotation.get("egoinvolve", ""),
            "source_startframe": annotation.get("startframe", ""),
            "youtube_id": annotation.get("youtubeID", ""),
            "width": width or "",
            "height": height or "",
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path) if do_hash else "",
        }

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(describe, images))

    columns = [
        "path", "video_id", "frame_no", "frame_index", "is_crash_frame",
        "timing", "weather", "egoinvolve", "source_startframe", "youtube_id",
        "width", "height", "bytes", "sha256",
    ]
    with (out_dir / "crashbest_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # 영상 단위 집계
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)

    video_rows = []
    for video_id, group in sorted(grouped.items()):
        group.sort(key=lambda r: r["frame_no"])
        crash_frames = [r["frame_index"] for r in group if r["is_crash_frame"] == 1]
        annotation = annotations.get(video_id, {})
        video_rows.append({
            "video_id": video_id,
            "frame_count": len(group),
            "first_crash_frame_index": crash_frames[0] if crash_frames else "",
            "crash_frame_count": len(crash_frames),
            "timing": annotation.get("timing", ""),
            "weather": annotation.get("weather", ""),
            "egoinvolve": annotation.get("egoinvolve", ""),
            "source_startframe": annotation.get("startframe", ""),
            "youtube_id": annotation.get("youtubeID", ""),
            "total_bytes": sum(r["bytes"] for r in group),
        })

    video_columns = [
        "video_id", "frame_count", "first_crash_frame_index", "crash_frame_count",
        "timing", "weather", "egoinvolve", "source_startframe", "youtube_id", "total_bytes",
    ]
    with (out_dir / "crashbest_videos.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=video_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(video_rows)

    resolutions = Counter(f"{r['width']}x{r['height']}" for r in rows)
    frame_counts = Counter(r["frame_count"] for r in video_rows)
    duplicates: dict[str, list[str]] = {}
    if do_hash:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            by_hash[row["sha256"]].append(Path(row["path"]).name)
        duplicates = {h: sorted(names) for h, names in by_hash.items() if len(names) > 1}

    return {
        "present": True,
        "image_count": len(rows),
        "video_count": len(video_rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "resolutions": dict(resolutions),
        "frames_per_video": {str(k): v for k, v in sorted(frame_counts.items())},
        "annotation_join_coverage": sum(1 for r in rows if r["timing"]) / max(1, len(rows)),
        "condition_distribution": {
            "timing": dict(Counter(r["timing"] for r in video_rows)),
            "weather": dict(Counter(r["weather"] for r in video_rows)),
            "egoinvolve": dict(Counter(r["egoinvolve"] for r in video_rows)),
        },
        "duplicate_groups": duplicates,
        "duplicate_image_count": sum(len(v) - 1 for v in duplicates.values()),
    }


# --------------------------------------------------------------------------- main scan
def build_files(root: Path, out_dir: Path, jobs: int, do_hash: bool) -> tuple[list[dict], list[dict]]:
    crashbest_abs = root / CRASHBEST_REL
    targets = [
        path for path in iter_files(root)
        if crashbest_abs not in path.parents and out_dir not in path.parents
    ]
    print(f"  일반 파일: {len(targets):,}개 스캔")

    def describe(path: Path) -> dict:
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        dataset, category = classify(rel, suffix)
        return {
            "path": rel,
            "dataset": dataset,
            "category": category,
            "extension": suffix,
            "bytes": path.stat().st_size,
            "mime": mimetypes.guess_type(path.name)[0] or "",
            "sha256": sha256_of(path) if do_hash else "",
        }

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(describe, targets))

    columns = ["path", "dataset", "category", "extension", "bytes", "mime", "sha256"]
    with (out_dir / "files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # 미디어 상세
    media_rows = []
    for row in rows:
        if row["category"] not in {"video", "image"}:
            continue
        path = root / row["path"]
        if row["category"] == "video":
            info = ffprobe(path)
            media_rows.append({
                "path": row["path"], "kind": "video", "bytes": row["bytes"],
                "codec": info.get("codec") or "", "container": info.get("container") or "",
                "width": info.get("width") or "", "height": info.get("height") or "",
                "r_frame_rate": info.get("r_frame_rate") or "",
                "avg_frame_rate": info.get("avg_frame_rate") or "",
                "container_duration_s": info.get("duration_s") or "",
                "decoded_frame_count": count_frames(path) or "",
                "pix_fmt": info.get("pix_fmt") or "",
            })
        else:
            width, height = image_size(path)
            media_rows.append({
                "path": row["path"], "kind": "image", "bytes": row["bytes"],
                "codec": "", "container": row["extension"].lstrip("."),
                "width": width or "", "height": height or "",
                "r_frame_rate": "", "avg_frame_rate": "", "container_duration_s": "",
                "decoded_frame_count": "", "pix_fmt": "",
            })

    media_columns = [
        "path", "kind", "bytes", "codec", "container", "width", "height",
        "r_frame_rate", "avg_frame_rate", "container_duration_s",
        "decoded_frame_count", "pix_fmt",
    ]
    with (out_dir / "media_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=media_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(media_rows)
    return rows, media_rows


# --------------------------------------------------------------------------- integrity
def duplicate_verdict(paths: list[str]) -> tuple[str, str] | None:
    """의도된 중복인지 판정. (severity, 이유) 또는 None(=예상 밖 중복)."""
    # Stage1 ORIGINAL 과 Stage2 영상은 동일한 CCD 원본 클립이다(data/SOURCES.md).
    if len(paths) == 2 and any("data/stage1/videos/original/" in p for p in paths) \
            and any("data/stage2/videos/" in p for p in paths):
        if Path(paths[0]).name == Path(paths[1]).name:
            return "ok", "설계상 동일: Stage1 ORIGINAL 과 Stage2 영상은 같은 CCD 원본 클립"
    # comma2k19 는 같은 클럭으로 샘플링된 센서들이 타임스탬프 배열을 공유한다.
    if all("/comma2k19/" in p and p.endswith("/t") for p in paths):
        return "ok", "설계상 동일: comma2k19 센서들이 동일 클럭 타임스탬프 배열을 공유"
    return None


def build_integrity(root: Path, files: list[dict], media: list[dict], crashbest: dict) -> dict:
    findings: list[dict] = []

    # 1) 동일 sha256 중복 — 의도된 중복과 낭비를 구분한다.
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in files:
        if row["sha256"]:
            by_hash[row["sha256"]].append(row["path"])
    duplicate_groups = {h: sorted(v) for h, v in by_hash.items() if len(v) > 1}
    expected, unexpected = {}, {}
    for digest, paths in sorted(duplicate_groups.items()):
        verdict = duplicate_verdict(paths)
        if verdict:
            expected[digest] = {"paths": paths, "reason": verdict[1]}
        else:
            unexpected[digest] = {"paths": paths}
            findings.append({
                "severity": "warning", "kind": "unexpected_duplicate",
                "detail": f"예상 밖 동일 내용 파일 {len(paths)}개 — 정리 대상",
                "paths": paths, "sha256": digest,
            })
    findings.append({
        "severity": "ok",
        "kind": "expected_duplicates",
        "detail": f"의도된 중복 {len(expected)}그룹 (삭제 대상 아님), 예상 밖 중복 {len(unexpected)}그룹",
        "groups": expected,
    })

    # 2) Stage1 클래스 누출 검사: original 과 rerecorded 가 같은 내용이면 학습 불가
    stage1 = {row["path"]: row["sha256"] for row in files if row["path"].startswith("data/stage1/videos/")}
    original = {Path(p).name: h for p, h in stage1.items() if "/original/" in p}
    rerecorded = {Path(p).name: h for p, h in stage1.items() if "/rerecorded/" in p}
    collisions = [name for name, h in original.items() if h and rerecorded.get(name) == h]
    findings.append({
        "severity": "error" if collisions else "ok",
        "kind": "stage1_class_collision",
        "detail": (
            f"ORIGINAL/RERECORDED 가 동일한 파일 {len(collisions)}건: {collisions}"
            if collisions else
            f"ORIGINAL {len(original)}건 / RERECORDED {len(rerecorded)}건 모두 내용 상이"
        ),
    })

    # 3) 필수 라벨 파일 존재 여부
    for stage, required in (("stage1", True), ("stage2", True), ("stage3", True)):
        path = root / "data" / stage / "labels.csv"
        findings.append({
            "severity": "ok" if path.is_file() else ("error" if required else "warning"),
            "kind": "labels_present",
            "detail": f"data/{stage}/labels.csv {'존재' if path.is_file() else '없음'}",
        })

    # 4) 라벨 ↔ 파일 참조 정합성
    for stage in ("stage1", "stage2"):
        labels = root / "data" / stage / "labels.csv"
        if not labels.is_file():
            continue
        missing = []
        with labels.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rel = row.get("path")
                if rel and not (root / "data" / stage / rel).is_file():
                    missing.append(rel)
        findings.append({
            "severity": "error" if missing else "ok",
            "kind": "labels_reference_integrity",
            "detail": f"data/{stage}/labels.csv 의 깨진 path {len(missing)}건" + (f": {missing}" if missing else ""),
        })

    # 5) Stage3 샘플링 주파수 vs 대회 스펙(10Hz) — 한 건으로 묶어 보고한다.
    stage3_media = [m for m in media if m["path"].startswith("data/stage3/videos/")]
    if stage3_media:
        findings.append({
            "severity": "warning",
            "kind": "stage3_frame_rate_mismatch",
            "detail": (
                "공개 예제 Stage3 영상은 comma2k19 원본 그대로 20Hz(패킷 간격 0.05s)이고 "
                "컨테이너는 40/1 로 잘못 선언되어 있다. 대회 평가 스펙은 10Hz "
                "(1프레임 = 1 sample = 0.1s)이므로 학습·검증 시 반드시 리샘플링해야 한다. "
                "또한 후반부 PTS 가 손상되어 컨테이너 duration 을 신뢰할 수 없다."
            ),
            "per_video": [
                {
                    "path": entry["path"],
                    "declared_r_frame_rate": entry.get("r_frame_rate") or "",
                    "container_duration_s": entry.get("container_duration_s") or "",
                    "decoded_frame_count": entry.get("decoded_frame_count") or "",
                }
                for entry in stage3_media
            ],
        })

    # 6) CrashBest 구조 계약
    if crashbest.get("present"):
        bad = {k: v for k, v in crashbest["frames_per_video"].items() if k != "50"}
        findings.append({
            "severity": "ok" if not bad else "error",
            "kind": "crashbest_frame_contract",
            "detail": f"영상당 50프레임 계약 {'충족' if not bad else f'위반: {bad}'} "
                      f"({crashbest['video_count']}개 영상 / {crashbest['image_count']}장)",
        })
        findings.append({
            "severity": "ok" if crashbest["annotation_join_coverage"] == 1.0 else "warning",
            "kind": "crashbest_annotation_join",
            "detail": f"Crash_Table.csv 조인 커버리지 {crashbest['annotation_join_coverage']:.4%}",
        })
        resolutions = crashbest["resolutions"]
        if len(resolutions) > 1:
            dominant = max(resolutions, key=resolutions.get)
            minority = {k: v for k, v in resolutions.items() if k != dominant}
            findings.append({
                "severity": "warning",
                "kind": "crashbest_resolution_mix",
                "detail": (
                    f"해상도가 균일하지 않다. 주 해상도 {dominant} "
                    f"{resolutions[dominant]:,}장 외에 {sum(minority.values()):,}장이 다른 해상도: "
                    f"{minority}. 전처리에서 letterbox/pillarbox 를 고려하지 않으면 "
                    "종횡비가 왜곡되고 Stage1 재녹화 판별에서 거짓 단서가 될 수 있다."
                ),
                "resolutions": resolutions,
            })
        if crashbest["duplicate_image_count"]:
            findings.append({
                "severity": "info",
                "kind": "crashbest_consecutive_duplicates",
                "detail": (
                    f"동일 내용 프레임 {crashbest['duplicate_image_count']}장 "
                    f"({len(crashbest['duplicate_groups'])}그룹). 원본 영상의 정지 구간에서 발생한 "
                    "연속 프레임 중복이며 삭제하면 50프레임 계약이 깨진다. 유지 권장."
                ),
                "groups": {h: v for h, v in sorted(crashbest["duplicate_groups"].items())},
            })

    counts = Counter(f["severity"] for f in findings)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {level: counts.get(level, 0) for level in ("error", "warning", "info", "ok")},
        "findings": findings,
        "duplicate_groups": {
            "expected": expected,
            "unexpected": unexpected,
            "crashbest_consecutive": crashbest.get("duplicate_groups", {}),
        },
    }


# --------------------------------------------------------------------------- catalog.json
def stage_schemas() -> dict:
    return {
        "stage1": {
            "task": "재녹화 여부 이진 분류",
            "weight": 0.2,
            "metric": "Macro-F1",
            "eval_input_layout": "data/stage1/videos/** (재귀 탐색, ID = 파일 stem)",
            "labels_csv": {"columns": ["ID", "path", "label"], "label_values": ["ORIGINAL", "RERECORDED"]},
            "submission_csv": {"columns": ["ID", "answer"], "answer_values": ["ORIGINAL", "RERECORDED"]},
        },
        "stage2": {
            "task": "충돌/진입 시점 + 회피공간·진입방향 분석",
            "weight": 0.4,
            "metric": "시각(초) 오차 + 범주 정확도",
            "eval_input_layout": "data/stage2/images/<ID>/frame_XXXXXX.jpg (ID = 폴더명, 프레임번호 = 파일명 숫자)",
            "labels_csv": {
                "columns": ["ID", "path", "t_collision", "t_entry", "evasion_space", "entry_side"],
                "notes": "-1 은 공개 정답 없음(손실 계산 제외). 공개 예제는 t_collision 만 유효.",
            },
            "submission_csv": {
                "columns": ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
                "evasion_space_values": [0, 1],
                "entry_side_values": ["LEFT", "RIGHT"],
            },
        },
        "stage3": {
            "task": "0.1초 단위 가감속·조향 범주 분류",
            "weight": 0.4,
            "metric": "범주 정확도 (STOPPED 프레임은 조향 평가 제외)",
            "eval_input_layout": "data/stage3/videos/*.mp4 (10Hz, ID = 파일 stem, 1프레임 = 1 sample = 0.1s)",
            "labels_csv": {
                "columns": ["ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"],
                "accel_values": list(("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")),
                "steer_values": list(("LEFT", "STRAIGHT", "RIGHT")),
            },
            "submission_csv": {"columns": ["ID", "sample_index", "accel_label", "steer_label"]},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out", type=Path, default=root / "catalog")
    parser.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 4) * 2))
    parser.add_argument("--skip-hash", action="store_true", help="sha256 계산 생략")
    args = parser.parse_args()

    do_hash = not args.skip_hash
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"카탈로그 생성 시작 (root={args.root})")

    files, media = build_files(args.root, args.out, args.jobs, do_hash)
    crashbest = build_crashbest(args.root, args.out, args.jobs, do_hash)
    integrity = build_integrity(args.root, files, media, crashbest)

    by_dataset: dict[str, dict] = defaultdict(lambda: {"file_count": 0, "bytes": 0, "categories": Counter()})
    for row in files:
        entry = by_dataset[row["dataset"]]
        entry["file_count"] += 1
        entry["bytes"] += row["bytes"]
        entry["categories"][row["category"]] += 1
    dataset_summary = {
        name: {
            "file_count": entry["file_count"],
            "bytes": entry["bytes"],
            "categories": dict(entry["categories"]),
        }
        for name, entry in sorted(by_dataset.items())
    }
    if crashbest.get("present"):
        dataset_summary["CCD"]["file_count"] += crashbest["image_count"]
        dataset_summary["CCD"]["bytes"] += crashbest["total_bytes"]
        dataset_summary["CCD"]["categories"]["image"] = (
            dataset_summary["CCD"]["categories"].get("image", 0) + crashbest["image_count"]
        )

    catalog = {
        "catalog_version": CATALOG_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hashes_included": do_hash,
        "competition": {
            "id": "236753",
            "name": "블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회",
            "url": "https://dacon.io/competitions/official/236753/overview/description",
            "hosts": ["행정안전부", "한국지능정보사회진흥원"],
            "organizer": "국립과학수사연구원",
            "operator": "데이콘",
            "submission_mode": "code (submit.zip)",
            "stage_weights": {"stage1": 0.2, "stage2": 0.4, "stage3": 0.4},
            "timeline": {
                "registration_start": "2026-08-18",
                "competition_start": "2026-08-26",
                "team_merge_deadline": "2026-09-23",
                "leaderboard_deadline": "2026-09-29",
                "competition_end": "2026-09-30",
                "round2_report_deadline": "2026-10-05",
                "final_results": "2026-10-16",
            },
            "eval_server": {
                "gpu": "NVIDIA L40S (44.7 GiB)",
                "cpu": "7 vCPU",
                "ram_gb": 60,
                "internet": False,
                "inference_time_limit_min": 60,
                "package_install_limit_min": 10,
                "submit_zip_limit_gb": 10,
                "uncompressed_limit_gb": 32,
                "pretrained_weights": "weights=None 필수 (인터넷 차단)",
            },
        },
        "stages": stage_schemas(),
        "index_files": {
            "catalog/files.csv": "CrashBest 이미지를 제외한 전체 파일 인덱스 (path, dataset, category, bytes, sha256)",
            "catalog/media_index.csv": "영상·이미지의 코덱/해상도/프레임수",
            "catalog/crashbest_index.csv": "CrashBest 75,000장 × 프레임 라벨 + 영상 메타 조인",
            "catalog/crashbest_videos.csv": "CrashBest 영상 단위 집계 (1,500행)",
            "catalog/integrity.json": "무결성 판정 결과",
        },
        "datasets": {
            "competition_samples": {
                "description": "대회 공개 예제. 평가서버와 동일한 Stage별 경로 구조로 배치되어 있다.",
                "root": "data/stage1, data/stage2, data/stage3",
                "provenance": "baseline.zip (대회 배포)",
                "license": "대회 참가 목적 사용",
            },
            "CCD": {
                "description": "Car Crash Dataset. 1,500개 크래시 영상의 50프레임 시퀀스 + 프레임별 크래시 주석.",
                "root": "data/external/CrashBest, data/external/Crash_Table.csv",
                "upstream": "https://github.com/Cogito2012/CarCrashDataset",
                "mirror": "https://www.kaggle.com/datasets/asefjamilajwad/car-crash-dataset-ccd",
                "paper": "Bao et al., Uncertainty-based Traffic Accident Anticipation, ACM MM 2020",
                "stats": {k: v for k, v in crashbest.items() if k != "duplicate_groups"},
                "used_for": ["stage1", "stage2"],
            },
            "comma2k19": {
                "description": "CA-280 고속도로 주행 데이터. Example 세그먼트에 CAN/IMU/GNSS/global_pose 포함.",
                "root": "data/external/comma2k19",
                "upstream": "https://github.com/commaai/comma2k19",
                "full_dataset": "Academic Torrents, ~100GB, 2019 segments, 33h+",
                "used_for": ["stage3"],
            },
        },
        "dataset_summary": dataset_summary,
        "totals": {
            "indexed_file_count": len(files) + crashbest.get("image_count", 0),
            "indexed_bytes": sum(r["bytes"] for r in files) + crashbest.get("total_bytes", 0),
        },
        "integrity_summary": integrity["summary"],
    }

    (args.out / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (args.out / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n생성 완료:")
    for name in sorted(p.name for p in args.out.iterdir() if p.is_file()):
        size = (args.out / name).stat().st_size
        print(f"  catalog/{name:26s} {size/1024:>10.1f} KB")
    print(f"\n무결성 요약: {integrity['summary']}")
    for finding in integrity["findings"]:
        if finding["severity"] in {"error", "warning"}:
            print(f"  [{finding['severity']:7s}] {finding['kind']}: {finding['detail']}")
    if integrity["summary"]["error"]:
        print(
            "\n  error 항목의 배경은 docs/05-data-integrity-report.md 에 있다.\n"
            "  이 스크립트는 '기록하는' 쪽이므로 판정 결과와 무관하게 0 을 반환한다.\n"
            "  실패로 취급할 게이트는 scripts/verify_integrity.py 다."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
