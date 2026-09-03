from __future__ import annotations

import pandas as pd

from config import TABLE_DIR
from src.plots import plot_prefix_scatter


def main():
    df = pd.read_csv(TABLE_DIR / "02_oof_prefix_predictions.csv")
    df["is_valuable"] = df["final_v002_purity"] >= df["good_threshold_train"]
    df["looks_poor_early"] = df["prediction"] <= df["poor_prediction_threshold_train"]
    df["late_bloomer"] = df["is_valuable"] & df["looks_poor_early"]

    rows = []
    for prefix, sub in df.groupby("prefix_fraction"):
        n_good = int(sub["is_valuable"].sum())
        n_late = int(sub["late_bloomer"].sum())
        rows.append(
            {
                "prefix_fraction": prefix,
                "n_runs": len(sub),
                "n_valuable": n_good,
                "n_late_bloomer_candidates": n_late,
                "late_bloomer_fraction_of_valuable": n_late / n_good if n_good else float("nan"),
            }
        )
        plot_prefix_scatter(sub, f"03_early_vs_final_prefix_{int(round(prefix*100)):02d}.png")

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "03_late_bloomer_summary.csv", index=False)
    df[df["late_bloomer"]].sort_values(["prefix_fraction", "final_v002_purity"], ascending=[True, False]).to_csv(
        TABLE_DIR / "03_late_bloomer_candidates.csv", index=False
    )
    print(summary.to_string(index=False))
    print("\nDefinition: valuable endpoint (>= training-fold 80th percentile) but early held-out prediction <= training-fold 30th percentile.")


if __name__ == "__main__":
    main()
