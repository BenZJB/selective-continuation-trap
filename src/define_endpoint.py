from __future__ import annotations

import numpy as np
import pandas as pd

from config import GC_ZIP, PROCESSED_DIR
from src.zipio import read_csv_member


def _last_available_row(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    if not columns:
        return None
    mask = df[columns].notna().any(axis=1)
    if not mask.any():
        return None
    return df.loc[mask].iloc[-1]


def endpoint_from_gc(df: pd.DataFrame) -> dict:
    """Exploratory chemistry endpoint.

    We use final V002 composition purity = max final mole fraction among components.
    This is a transparent proxy for separation quality, not a definitive process-economic
    objective. A later paper stage should compare purity, recovery, and cost-aware utility.
    """
    v002_cols = [c for c in df.columns if c.endswith("_mol_V002")]
    v001_cols = [c for c in df.columns if c.endswith("_mol_V001")]

    out = {
        "final_v002_purity": np.nan,
        "final_v002_component": None,
        "final_v001_purity": np.nan,
        "final_v001_component": None,
        "gc_rows": len(df),
    }

    r2 = _last_available_row(df, v002_cols)
    if r2 is not None:
        vals = pd.to_numeric(r2[v002_cols], errors="coerce")
        if vals.notna().any():
            col = vals.idxmax()
            out["final_v002_purity"] = float(vals[col])
            out["final_v002_component"] = col.removesuffix("_mol_V002")

    r1 = _last_available_row(df, v001_cols)
    if r1 is not None:
        vals = pd.to_numeric(r1[v001_cols], errors="coerce")
        if vals.notna().any():
            col = vals.idxmax()
            out["final_v001_purity"] = float(vals[col])
            out["final_v001_component"] = col.removesuffix("_mol_V001")

    return out


def build_endpoints(run_table: pd.DataFrame | None = None) -> pd.DataFrame:
    if run_table is None:
        run_table = pd.read_csv(PROCESSED_DIR / "run_table.csv")

    rows = []
    for _, row in run_table[run_table["has_gc_timeseries"] == True].iterrows():  # noqa: E712
        gc = read_csv_member(GC_ZIP, row["gc_member"])
        vals = endpoint_from_gc(gc)
        rows.append(
            {
                "run_id": row["run_id"],
                "mixture": row["mixture"],
                "operating_point": row["operating_point"],
                **vals,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "endpoints.csv", index=False)
    return out


if __name__ == "__main__":
    df = build_endpoints()
    print(f"Wrote {len(df)} GC endpoints to {PROCESSED_DIR / 'endpoints.csv'}")
    print(df.groupby("mixture")["final_v002_purity"].agg(["count", "mean", "std", "min", "max"]).to_string())
