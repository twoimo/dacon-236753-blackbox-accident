#!/usr/bin/env python3
"""data/stage2/labels.csv 재생성.

분실된 대회 배포 labels.csv 를 CCD 공식 주석(data/external/Crash_Table.csv)에서
결정론적으로 복원한다. 복원 규칙은 data/SOURCES.md 에 명시된 원 생성 규칙과 동일하다.

  t_collision = Crash_Table.csv 의 frame_1..frame_50 중 첫 번째 값이 1 인 프레임의
                0-기반 인덱스 (= CCD Crash-1500.txt 의 first positive frame)

t_entry / evasion_space / entry_side 는 공개 정답이 존재하지 않으므로 -1 로 둔다.
베이스라인 학습 코드는 -1 항목을 손실 계산에서 제외한다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

STAGE2_COLUMNS = ["ID", "path", "t_collision", "t_entry", "evasion_space", "entry_side"]
NO_LABEL = -1


def first_positive_frame(row: dict[str, str]) -> int | None:
    """frame_1..frame_50 중 첫 positive 프레임의 0-기반 인덱스."""
    for index in range(50):
        if row[f"frame_{index + 1}"].strip() == "1":
            return index
    return None


def build_rows(video_ids: list[str], crash_table: Path) -> list[dict[str, object]]:
    with crash_table.open(newline="", encoding="utf-8") as handle:
        annotations = {row["vidname"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    for video_id in video_ids:
        annotation = annotations.get(video_id)
        if annotation is None:
            raise SystemExit(f"Crash_Table.csv 에 {video_id} 주석이 없습니다.")
        collision = first_positive_frame(annotation)
        if collision is None:
            raise SystemExit(f"{video_id} 에 positive 프레임이 없습니다.")
        rows.append(
            {
                "ID": video_id,
                "path": f"videos/{video_id}.mp4",
                "t_collision": collision,
                "t_entry": NO_LABEL,
                "evasion_space": NO_LABEL,
                "entry_side": NO_LABEL,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage2 = args.root / "data" / "stage2"
    crash_table = args.root / "data" / "external" / "Crash_Table.csv"
    if not crash_table.is_file():
        raise SystemExit(f"필수 파일 없음: {crash_table}")

    video_ids = sorted(p.stem for p in (stage2 / "videos").glob("*.mp4"))
    if not video_ids:
        raise SystemExit(f"영상이 없습니다: {stage2 / 'videos'}")

    rows = build_rows(video_ids, crash_table)
    for row in rows:
        print(f"  {row['ID']}: t_collision={row['t_collision']}")

    if args.dry_run:
        print("dry-run: 파일을 쓰지 않았습니다.")
        return 0

    target = stage2 / "labels.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"생성 완료: {target.relative_to(args.root)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
