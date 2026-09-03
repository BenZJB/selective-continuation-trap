from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR, PRIMARY_MIXTURE, PROCESSED_DIR, TABLE_DIR


def main():
    endpoints = pd.read_csv(PROCESSED_DIR / "endpoints.csv")
    print("GC endpoint coverage by mixture:")
    print(endpoints.groupby("mixture")["final_v002_purity"].agg(["count", "mean", "std", "min", "max"]).to_string())

    primary = endpoints[endpoints["mixture"] == PRIMARY_MIXTURE].dropna(subset=["final_v002_purity"]).copy()
    primary.sort_values("final_v002_purity", ascending=False).to_csv(TABLE_DIR / "primary_mixture_endpoints_ranked.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(primary["final_v002_purity"], bins=15)
    ax.set_xlabel("Final V002 purity (proxy endpoint)")
    ax.set_ylabel("Number of experiments")
    ax.set_title(f"Endpoint distribution: {PRIMARY_MIXTURE}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_endpoint_distribution.png", dpi=200)
    plt.close(fig)

    print(f"\nPrimary mixture usable endpoints: {len(primary)}")
    print("IMPORTANT: final V002 purity is an exploratory proxy, not yet the final process-economic objective.")


if __name__ == "__main__":
    main()
