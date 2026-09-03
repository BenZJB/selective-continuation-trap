from __future__ import annotations

import numpy as np
import pandas as pd

from config import ACTUATORS_ZIP, PREFIXES, PROCESSED_DIR, SENSORS_ZIP
from src.zipio import read_csv_member


def _numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.lower() == "time"]
    x = df.drop(columns=drop, errors="ignore").copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _slope(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return np.nan
    y = arr[mask]
    x = np.flatnonzero(mask).astype(float)
    # Units are approximately change per second because the raw data are ~1 Hz.
    return float(np.polyfit(x, y, deg=1)[0])


def summarize(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    x = _numeric_data(df)
    feats: dict[str, float] = {}
    for c in x.columns:
        s = x[c]
        valid = s.dropna()
        if len(valid) == 0:
            stats = {k: np.nan for k in ("last", "mean", "std", "min", "max", "delta", "slope")}
        else:
            stats = {
                "last": float(valid.iloc[-1]),
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=0)),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "delta": float(valid.iloc[-1] - valid.iloc[0]),
                "slope": _slope(s),
            }
        for k, v in stats.items():
            feats[f"{prefix}__{c}__{k}"] = v
    return feats


def _prefix_rows(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    n = max(2, int(np.ceil(len(df) * fraction)))
    n = min(n, len(df))
    return df.iloc[:n].copy()


def build_prefix_features(run_table: pd.DataFrame | None = None, endpoints: pd.DataFrame | None = None) -> pd.DataFrame:
    if run_table is None:
        run_table = pd.read_csv(PROCESSED_DIR / "run_table.csv")
    if endpoints is None:
        endpoints = pd.read_csv(PROCESSED_DIR / "endpoints.csv")

    eligible = run_table.merge(endpoints, on=["run_id", "mixture", "operating_point"], how="inner")
    eligible = eligible[eligible["has_sensor"] & eligible["has_actuator"]].copy()

    rows = []
    for _, r in eligible.iterrows():
        sensors = read_csv_member(SENSORS_ZIP, r["sensor_member"])
        actuators = read_csv_member(ACTUATORS_ZIP, r["actuator_member"])
        n = min(len(sensors), len(actuators))
        sensors = sensors.iloc[:n]
        actuators = actuators.iloc[:n]

        for frac in PREFIXES:
            sf = _prefix_rows(sensors, frac)
            af = _prefix_rows(actuators, frac)
            feats = {}
            feats.update(summarize(sf, "sensor"))
            feats.update(summarize(af, "actuator"))
            rows.append(
                {
                    "run_id": r["run_id"],
                    "mixture": r["mixture"],
                    "operating_point": r["operating_point"],
                    "experiment": r["experiment"],
                    "prefix_fraction": frac,
                    "prefix_rows": min(len(sf), len(af)),
                    "duration_minutes": r["duration_minutes"],
                    "final_v002_purity": r["final_v002_purity"],
                    **feats,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "prefix_features.csv", index=False)
    return out


if __name__ == "__main__":
    df = build_prefix_features()
    print(f"Wrote {len(df)} prefix samples ({df['run_id'].nunique()} physical runs) to {PROCESSED_DIR / 'prefix_features.csv'}")
    print(df.groupby(["mixture", "prefix_fraction"])["run_id"].nunique().to_string())
