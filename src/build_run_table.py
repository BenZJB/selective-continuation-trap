from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ACTUATORS_ZIP,
    GC_ZIP,
    METADATA_ZIP,
    OPLOGS_ZIP,
    PROCESSED_DIR,
    SENSORS_ZIP,
)
from src.zipio import build_member_index, parse_run_key, read_csv_member


def _duration_seconds_from_time_column(df: pd.DataFrame, column: str = "Time") -> float:
    if column not in df.columns or len(df) < 2:
        return float(max(len(df) - 1, 0))
    # Dataset is approximately 1 Hz. Parsing handles midnight rollover if needed.
    t = pd.to_datetime(df[column].astype(str), format="%H:%M:%S", errors="coerce")
    if t.notna().sum() < 2:
        return float(max(len(df) - 1, 0))
    secs = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    # Repair a possible midnight rollover.
    secs = np.where(secs < -43200, secs + 86400, secs)
    valid = secs[np.isfinite(secs)]
    if len(valid) == 0:
        return float(max(len(df) - 1, 0))
    duration = float(valid[-1])
    if duration <= 0:
        duration = float(max(len(df) - 1, 0))
    return duration


def _metadata_diagnostics(member: str) -> tuple[int, float | None]:
    df = read_csv_member(METADATA_ZIP, member)
    if "Label (anomaly)" not in df.columns:
        return 0, None
    lab = pd.to_numeric(df["Label (anomaly)"], errors="coerce").fillna(0).to_numpy()
    idx = np.flatnonzero(lab != 0)
    if len(idx) == 0:
        return 0, None
    return int(len(idx)), float(idx[0] / max(len(df) - 1, 1))


def build_run_table() -> pd.DataFrame:
    sensor_idx = build_member_index(SENSORS_ZIP, required_segment="Operation")
    actuator_idx = build_member_index(ACTUATORS_ZIP, required_segment="Operation")
    metadata_idx = build_member_index(METADATA_ZIP, required_segment="Operation")
    gc_idx = build_member_index(GC_ZIP, required_segment="timeseries")
    oplog_idx = build_member_index(OPLOGS_ZIP)

    all_ids = sorted(set(sensor_idx) | set(actuator_idx) | set(metadata_idx) | set(gc_idx))
    rows = []
    for run_id in all_ids:
        source_member = sensor_idx.get(run_id) or actuator_idx.get(run_id) or metadata_idx.get(run_id) or gc_idx.get(run_id)
        key = parse_run_key(source_member)
        sensor_member = sensor_idx.get(run_id)
        n_sensor_rows = np.nan
        duration_s = np.nan
        if sensor_member:
            sdf = read_csv_member(SENSORS_ZIP, sensor_member)
            n_sensor_rows = len(sdf)
            duration_s = _duration_seconds_from_time_column(sdf, "Time")

        anomaly_count, first_anomaly_fraction = (0, None)
        if run_id in metadata_idx:
            anomaly_count, first_anomaly_fraction = _metadata_diagnostics(metadata_idx[run_id])

        rows.append(
            {
                "run_id": run_id,
                "mixture": key.mixture,
                "operating_point": key.operating_point,
                "experiment": key.experiment,
                "split_label": "train" if key.experiment.startswith("train_") else "test",
                "normality_label": "normal" if "_normal_" in key.experiment and "anormal" not in key.experiment else "anormal",
                "sensor_member": sensor_member,
                "actuator_member": actuator_idx.get(run_id),
                "metadata_member": metadata_idx.get(run_id),
                "gc_member": gc_idx.get(run_id),
                "oplog_member": oplog_idx.get(run_id),
                "has_sensor": sensor_member is not None,
                "has_actuator": run_id in actuator_idx,
                "has_metadata": run_id in metadata_idx,
                "has_gc_timeseries": run_id in gc_idx,
                "has_oplog": run_id in oplog_idx,
                "n_sensor_rows": n_sensor_rows,
                "duration_seconds": duration_s,
                "duration_minutes": duration_s / 60.0 if pd.notna(duration_s) else np.nan,
                "anomaly_points": anomaly_count,
                "first_anomaly_fraction": first_anomaly_fraction,
            }
        )

    out = pd.DataFrame(rows).sort_values(["mixture", "operating_point", "experiment"]).reset_index(drop=True)
    path = PROCESSED_DIR / "run_table.csv"
    out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    df = build_run_table()
    print(f"Wrote {len(df)} runs to {PROCESSED_DIR / 'run_table.csv'}")
    print(df.groupby("mixture").agg(runs=("run_id", "size"), gc_runs=("has_gc_timeseries", "sum")).to_string())
