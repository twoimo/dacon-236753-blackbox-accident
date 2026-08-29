#!/usr/bin/env python3
"""제출 전 검증 — docs/03 체크리스트를 자동 점검한다.

근거: docs/03-evaluation-and-submission.md §8, research/synthesis/.
사용: python scripts/validate_submission.py [submit.zip 또는 스테이지 폴더]
성공 시 exit 0 + METRIC submission_valid=1.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REQUIRED_TOP = {"model", "inference.py", "requirements.txt"}
REQUIRED_FUNCS = ["predict_stage1", "predict_stage2", "predict_stage3"]


def _names_from_zip(zpath: Path) -> list[str]:
    with zipfile.ZipFile(zpath) as z:
        return z.namelist()


def validate_zip(zpath: Path) -> list[str]:
    problems = []
    names = _names_from_zip(zpath)
    tops = {n.split("/")[0] for n in names if n.strip()}
    for req in REQUIRED_TOP:
        if req not in tops:
            problems.append(f"최상위에 {req} 없음")
    # inference.py 안에 필수 함수가 정의/노출되는지 (텍스트 검사)
    with zipfile.ZipFile(zpath) as z:
        if "inference.py" in names:
            body = z.read("inference.py").decode("utf-8", "ignore")
            for fn in REQUIRED_FUNCS:
                if fn not in body:
                    problems.append(f"inference.py 에 {fn} 노출 안 됨")
    # 10GB 제한
    if zpath.stat().st_size > 10 * 1024**3:
        problems.append("submit.zip 10GB 초과")
    return problems


def validate_repo() -> list[str]:
    """zip 없이 저장소 상태로 사전 점검 (inference.py + src 존재/함수 노출)."""
    problems = []
    root = Path(__file__).resolve().parents[1]
    inf = root / "inference.py"
    if not inf.exists():
        problems.append("inference.py 없음")
    else:
        body = inf.read_text(encoding="utf-8", errors="ignore")
        for fn in REQUIRED_FUNCS:
            if fn not in body:
                problems.append(f"inference.py 에 {fn} 노출 안 됨")
    for stage in ("stage1", "stage2", "stage3"):
        if not (root / "src" / stage / "predict.py").exists():
            problems.append(f"src/{stage}/predict.py 없음")
    return problems


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and Path(target).suffix == ".zip":
        problems = validate_zip(Path(target))
        mode = f"zip:{target}"
    else:
        problems = validate_repo()
        mode = "repo"
    print(f"[validate] 대상={mode}")
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        print("METRIC submission_valid=0")
        return 1
    print("  [ok] 필수 구성/함수/용량 규격 통과")
    print("METRIC submission_valid=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
