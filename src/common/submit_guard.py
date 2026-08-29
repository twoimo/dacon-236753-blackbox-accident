"""제출 규격 방어 유틸 — docs/03-evaluation-and-submission.md 채점 함정 대응.

제출 오류는 제출 횟수를 차감하므로, 반환 직전 반드시 이 함수들로 검증/정제한다.
근거: docs/03 §채점, research/02-stage2-anticipation/README.md (후처리)
"""
from __future__ import annotations

import pandas as pd

STAGE1_COLS = ["ID", "answer"]
STAGE2_COLS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
STAGE3_COLS = ["ID", "sample_index", "accel_label", "steer_label"]

ANSWER_S1 = {"ORIGINAL", "RERECORDED"}
ACCEL_S3 = {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}
STEER_S3 = {"LEFT", "STRAIGHT", "RIGHT"}
SIDE_S2 = {"LEFT", "RIGHT"}


class SubmitFormatError(ValueError):
    """제출 규격 위반. 반환 전에 잡아 고쳐야 한다."""


def check_stage1(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != STAGE1_COLS:
        raise SubmitFormatError(f"Stage1 컬럼 불일치: {list(df.columns)} != {STAGE1_COLS}")
    bad = set(df["answer"].unique()) - ANSWER_S1
    if bad:
        raise SubmitFormatError(f"Stage1 허용 외 answer: {bad}")
    if df["ID"].isna().any():
        raise SubmitFormatError("Stage1 ID 결측")
    return df


def check_stage2(df: pd.DataFrame, frame_counts: dict[str, int] | None = None) -> pd.DataFrame:
    """frame_counts: {ID: 프레임수}. 주면 범위 초과/음수 프레임을 clamp."""
    if list(df.columns) != STAGE2_COLS:
        raise SubmitFormatError(f"Stage2 컬럼 불일치: {list(df.columns)} != {STAGE2_COLS}")
    df = df.copy()
    for col in ["collision_frame", "entry_frame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise SubmitFormatError(f"Stage2 {col} 비수치/결측 존재")
        df[col] = df[col].round().astype(int).clip(lower=0)
        if frame_counts is not None:
            df[col] = df.apply(
                lambda r, c=col: min(int(r[c]), frame_counts.get(str(r["ID"]), r[c]) - 1),
                axis=1,
            )
    df["evasion_space"] = pd.to_numeric(df["evasion_space"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    bad_side = set(df["entry_side"].unique()) - SIDE_S2
    if bad_side:
        raise SubmitFormatError(f"Stage2 허용 외 entry_side: {bad_side}")
    return df


def check_stage3(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != STAGE3_COLS:
        raise SubmitFormatError(f"Stage3 컬럼 불일치: {list(df.columns)} != {STAGE3_COLS}")
    df = df.copy()
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    if df["sample_index"].isna().any():
        raise SubmitFormatError("Stage3 sample_index 비수치/결측")
    df["sample_index"] = df["sample_index"].astype(int)
    bad_a = set(df["accel_label"].unique()) - ACCEL_S3
    bad_s = set(df["steer_label"].unique()) - STEER_S3
    if bad_a:
        raise SubmitFormatError(f"Stage3 허용 외 accel_label: {bad_a}")
    if bad_s:
        raise SubmitFormatError(f"Stage3 허용 외 steer_label: {bad_s}")
    # STOPPED 프레임도 steer_label 이 있어야 한다 (docs/03) — 결측이면 STRAIGHT 채움
    if df["steer_label"].isna().any():
        df["steer_label"] = df["steer_label"].fillna("STRAIGHT")
    return df


if __name__ == "__main__":
    # 자기 테스트: 규격 위반을 실제로 잡는지 확인
    import sys

    ok = True
    try:
        check_stage1(pd.DataFrame({"ID": ["a"], "answer": ["MAYBE"]}))
        ok = False
        print("FAIL: Stage1 잘못된 값 통과됨")
    except SubmitFormatError:
        print("OK: Stage1 잘못된 answer 차단")
    d3 = pd.DataFrame({"ID": ["v"], "sample_index": [0], "accel_label": ["CONSTANT"], "steer_label": [None]})
    fixed = check_stage3(d3)
    assert fixed["steer_label"].iloc[0] == "STRAIGHT", "STOPPED steer 채움 실패"
    print("OK: Stage3 결측 steer_label → STRAIGHT 채움")
    print("METRIC guard_selftest=" + ("1" if ok else "0"))
    sys.exit(0 if ok else 1)
