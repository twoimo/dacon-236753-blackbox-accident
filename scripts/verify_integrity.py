#!/usr/bin/env python3
"""catalog/ 의 기록과 실제 파일을 대조해 무결성을 검증한다.

build_catalog.py 가 "현재 상태를 기록"하는 쪽이라면, 이 스크립트는 "기록과
현재가 같은지 확인"하는 쪽이다. 데이터를 구글 드라이브에서 내려받은 직후,
또는 브랜치를 갈아탄 뒤 실행한다.

  python scripts/verify_integrity.py            # 크기만 대조 (빠름)
  python scripts/verify_integrity.py --hash     # sha256 까지 대조 (느림)
  python scripts/verify_integrity.py --hash --sample 500

종료 코드: 0 = 일치, 1 = 불일치 발견
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(root: Path, rows: list[dict[str, str]], do_hash: bool, jobs: int) -> list[str]:
    problems: list[str] = []

    def verify(row: dict[str, str]) -> str | None:
        path = root / row["path"]
        if not path.is_file():
            return f"결측: {row['path']}"
        actual = path.stat().st_size
        expected = int(row["bytes"])
        if actual != expected:
            return f"크기 불일치: {row['path']} (기록 {expected} / 실제 {actual})"
        if do_hash and row.get("sha256"):
            digest = sha256_of(path)
            if digest != row["sha256"]:
                return f"해시 불일치: {row['path']}"
        return None

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        problems.extend(result for result in pool.map(verify, rows) if result)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--catalog", type=Path, default=root / "catalog")
    parser.add_argument("--hash", action="store_true", help="sha256 까지 대조")
    parser.add_argument("--sample", type=int, default=0, help="CrashBest 를 N장만 표본 검사 (0=전수)")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    if not (args.catalog / "catalog.json").is_file():
        raise SystemExit(f"카탈로그가 없습니다. 먼저 build_catalog.py 를 실행하세요: {args.catalog}")

    catalog = json.loads((args.catalog / "catalog.json").read_text(encoding="utf-8"))
    print(f"카탈로그 v{catalog['catalog_version']} ({catalog['generated_at']}) 기준 검증")
    if args.hash and not catalog.get("hashes_included"):
        print("  주의: 카탈로그가 --skip-hash 로 생성되어 해시 대조를 건너뜁니다.")
        args.hash = False

    problems: list[str] = []

    general = load_index(args.catalog / "files.csv")
    print(f"  일반 파일 {len(general):,}개 검증 (hash={'on' if args.hash else 'off'})")
    problems += check(args.root, general, args.hash, args.jobs)

    images = load_index(args.catalog / "crashbest_index.csv")
    if images:
        target = images
        if args.sample and args.sample < len(images):
            target = random.Random(20260828).sample(images, args.sample)
            print(f"  CrashBest {len(target):,}/{len(images):,}장 표본 검증")
        else:
            print(f"  CrashBest {len(target):,}장 전수 검증")
        problems += check(args.root, target, args.hash, args.jobs)

    # 라벨 파일 존재 여부. stage3 는 복구 불가로 문서화된 기존 결손이므로
    # 게이트를 실패시키지 않고 참고 사항으로만 알린다(docs/05-data-integrity-report.md 2절).
    notes: list[str] = []
    for stage in ("stage1", "stage2", "stage3"):
        labels = args.root / "data" / stage / "labels.csv"
        if labels.is_file():
            continue
        if stage == "stage3":
            notes.append(
                "data/stage3/labels.csv 없음 — 복구 불가로 문서화된 기존 결손. "
                "대체: data/stage3/labels_comma2k19.csv "
                "(근거: docs/05-data-integrity-report.md 2절)"
            )
        else:
            problems.append(f"필수 라벨 결측: data/{stage}/labels.csv")

    print()
    if notes:
        print("참고:")
        for note in notes:
            print(f"  - {note}")
        print()
    if problems:
        print(f"불일치 {len(problems)}건:")
        for problem in problems[:50]:
            print(f"  - {problem}")
        if len(problems) > 50:
            print(f"  ... 외 {len(problems) - 50}건")
        return 1
    print("모든 항목이 카탈로그와 일치합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
