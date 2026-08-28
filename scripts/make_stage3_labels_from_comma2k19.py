#!/usr/bin/env python3
"""comma2k19 CAN 로그에서 Stage 3 가감속·조향 라벨을 생성한다.

대회 공개 예제(data/stage3/videos/OPEN_00*.mp4)의 labels.csv 는 로컬에 존재하지
않으며, 해당 영상이 comma2k19 의 어느 세그먼트에서 나왔는지 알 수 없으므로
복원할 수 없다(추측값을 만들지 않는다). 대신 이 스크립트는 CAN 정답을 가진
comma2k19 세그먼트에 대해 동일 스키마의 라벨을 생성하여, Stage 3 학습 데이터를
직접 확장할 수 있게 한다.

라벨 정의
  accel_label : 종방향 속도 미분(dv/dt) 기준
      STOPPED       speed  < --stop-speed
      ACCELERATING  dv/dt >= +--accel-threshold
      DECELERATING  dv/dt <= -–accel-threshold
      CONSTANT      그 외
  steer_label : CAN steering_angle(도) 기준
      LEFT      angle >= +--steer-threshold
      RIGHT     angle <= -–steer-threshold
      STRAIGHT  그 외

조향 부호 규약은 가정하지 않고 매 실행마다 global_pose 로부터 유도한 ENU
헤딩 변화율(yaw rate)과의 상관계수로 검증한다. openpilot 규약대로 양수 = 좌회전
이면 상관계수가 음수(ENU atan2(E, N) 에서 양의 방향은 시계방향)로 나온다.
검증에 실패하면 종료하여 부호가 뒤집힌 라벨이 생성되는 것을 막는다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ACCEL_LABELS = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_LABELS = ("LEFT", "STRAIGHT", "RIGHT")
STAGE3_COLUMNS = ["ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"]


def load(segment: Path, relative: str) -> np.ndarray:
    path = segment / relative
    if not path.is_file():
        raise SystemExit(f"필수 센서 파일 없음: {path}")
    return np.load(path)


def enu_yaw_rate(segment: Path, frame_times: np.ndarray) -> np.ndarray:
    """ECEF 위치·속도에서 지역 ENU 평면상의 헤딩 변화율(rad/s)을 계산."""
    positions = load(segment, "global_pose/frame_positions")
    velocities = load(segment, "global_pose/frame_velocities")
    center = positions.mean(axis=0)
    up = center / np.linalg.norm(center)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    heading = np.unwrap(np.arctan2(velocities @ east, velocities @ north))
    return np.gradient(heading, frame_times)


def verify_steering_sign(steer: np.ndarray, yaw_rate: np.ndarray, speed: np.ndarray) -> float:
    moving = speed > 2.0
    if moving.sum() < 50:
        raise SystemExit("주행 구간이 너무 짧아 조향 부호를 검증할 수 없습니다.")
    correlation = float(np.corrcoef(steer[moving], yaw_rate[moving])[0, 1])
    if not np.isfinite(correlation):
        raise SystemExit("조향 부호 검증 실패: 상관계수가 유한하지 않습니다.")
    if correlation > 0:
        raise SystemExit(
            "조향 부호 검증 실패: steering_angle 양수가 우회전에 대응합니다 "
            f"(corr={correlation:+.3f}). 이 세그먼트는 openpilot 규약과 다릅니다."
        )
    return correlation


def classify(
    speed: np.ndarray,
    accel: np.ndarray,
    steer: np.ndarray,
    stop_speed: float,
    accel_threshold: float,
    steer_threshold: float,
) -> tuple[list[str], list[str]]:
    accel_labels = np.where(
        speed < stop_speed,
        "STOPPED",
        np.where(
            accel >= accel_threshold,
            "ACCELERATING",
            np.where(accel <= -accel_threshold, "DECELERATING", "CONSTANT"),
        ),
    )
    steer_labels = np.where(
        steer >= steer_threshold,
        "LEFT",
        np.where(steer <= -steer_threshold, "RIGHT", "STRAIGHT"),
    )
    return accel_labels.tolist(), steer_labels.tolist()


def process(segment: Path, video_id: str, args: argparse.Namespace) -> list[dict[str, object]]:
    frame_times = load(segment, "global_pose/frame_times")
    speed_t = load(segment, "processed_log/CAN/speed/t")
    speed_v = load(segment, "processed_log/CAN/speed/value")
    speed_v = speed_v[:, 0] if speed_v.ndim == 2 else speed_v
    steer_t = load(segment, "processed_log/CAN/steering_angle/t")
    steer_v = load(segment, "processed_log/CAN/steering_angle/value")
    steer_v = steer_v[:, 0] if steer_v.ndim == 2 else steer_v

    # CAN 은 ~83Hz, 영상은 20Hz 이므로 영상 프레임 시각으로 보간한다.
    speed = np.interp(frame_times, speed_t, speed_v)
    steer = np.interp(frame_times, steer_t, steer_v)
    accel = np.gradient(speed, frame_times)

    correlation = verify_steering_sign(steer, enu_yaw_rate(segment, frame_times), speed)
    video_hz = 1.0 / float(np.median(np.diff(frame_times)))
    stride = max(1, round(video_hz / args.label_hz))
    print(
        f"  {video_id}: {len(frame_times)} frames @ {video_hz:.1f}Hz -> "
        f"{args.label_hz:.0f}Hz stride={stride}, 조향부호 corr={correlation:+.3f}"
    )

    indices = np.arange(0, len(frame_times), stride)
    accel_labels, steer_labels = classify(
        speed[indices], accel[indices], steer[indices],
        args.stop_speed, args.accel_threshold, args.steer_threshold,
    )
    origin = frame_times[0]
    return [
        {
            "ID": video_id,
            "sample_index": sample_index,
            "frame_index": int(frame_index),
            "time_seconds": round(float(frame_times[frame_index] - origin), 3),
            "accel_label": accel_label,
            "steer_label": steer_label,
        }
        for sample_index, (frame_index, accel_label, steer_label) in enumerate(
            zip(indices, accel_labels, steer_labels)
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out", type=Path, default=root / "data" / "stage3" / "labels_comma2k19.csv")
    parser.add_argument("--label-hz", type=float, default=10.0, help="라벨 샘플링 주파수 (대회 스펙 10Hz)")
    parser.add_argument("--stop-speed", type=float, default=0.5, help="STOPPED 판정 속도 (m/s)")
    parser.add_argument("--accel-threshold", type=float, default=0.3, help="가감속 판정 임계 (m/s^2)")
    parser.add_argument("--steer-threshold", type=float, default=1.0, help="조향 판정 임계 (deg)")
    args = parser.parse_args()

    comma_root = args.root / "data" / "external" / "comma2k19"
    segments = sorted(
        path.parent
        for path in comma_root.rglob("global_pose/frame_times")
        if (path.parent.parent / "processed_log" / "CAN" / "speed" / "value").is_file()
    )
    segments = [segment.parent for segment in segments]
    if not segments:
        raise SystemExit(f"comma2k19 세그먼트를 찾지 못했습니다: {comma_root}")

    rows: list[dict[str, object]] = []
    for segment in segments:
        # 예: <route>|<timestamp>/40  ->  C2K19_<route>_<timestamp>_40
        route, _, stamp = segment.parent.name.partition("|")
        video_id = f"C2K19_{route}_{stamp}_{segment.name}".replace("--", "-")
        rows.extend(process(segment, video_id, args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE3_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    print(f"생성 완료: {args.out.relative_to(args.root)} ({len(rows)} rows, {len(segments)} segments)")
    print("  accel 분포:", dict(Counter(r["accel_label"] for r in rows)))
    print("  steer 분포:", dict(Counter(r["steer_label"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
