from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")
    return pd.read_csv(path)


def save_summary_table_figure(df: pd.DataFrame, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 1:
# Recall gap vs stopping aggressiveness
# ============================================================

def plot_recall_gap_vs_threshold():
    df = load_csv("05_stop_sweep_summary.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    x = df["stop_quantile"]
    y = df["valuable_run_recall_delta_mean"]
    y_low = df["valuable_run_recall_delta_lower"]
    y_high = df["valuable_run_recall_delta_upper"]

    ax.axhline(0, linewidth=1)
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y_low, y_high, alpha=0.18)

    ax.set_xlabel("Stopping threshold quantile")
    ax.set_ylabel("Recall gap (completed-only − oracle)")
    ax.set_title("Selective-censoring penalty on valuable-run recall")

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "supervisor_01_recall_gap_vs_threshold.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# FIGURE 2:
# Valuable false-stop gap vs stopping aggressiveness
# ============================================================

def plot_false_stop_gap_vs_threshold():
    df = load_csv("05_stop_sweep_summary.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    x = df["stop_quantile"]
    y = df["valuable_false_stop_rate_delta_mean"]
    y_low = df["valuable_false_stop_rate_delta_lower"]
    y_high = df["valuable_false_stop_rate_delta_upper"]

    ax.axhline(0, linewidth=1)
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y_low, y_high, alpha=0.18)

    ax.set_xlabel("Stopping threshold quantile")
    ax.set_ylabel("False-stop gap (completed-only − oracle)")
    ax.set_title("Selective-censoring penalty on killing valuable runs")

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "supervisor_02_false_stop_gap_vs_threshold.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# FIGURE 3:
# Cumulative valuable-run recall over rounds
# Uses q = 0.40 by default because that was quite illustrative
# ============================================================

def plot_cumulative_recall_over_rounds(
    stop_quantile: float = 0.40,
    max_round: int = 5
):
    df = load_csv("05_cumulative_round_metrics.csv")

    # Select stopping threshold
    df = df[df["stop_quantile"] == stop_quantile].copy()

    # IMPORTANT:
    # Exclude later rounds that are not represented by the same set
    # of deployment seeds.
    df = df[df["round"] <= max_round].copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for learner, sub in df.groupby("learner"):

        agg = (
            sub.groupby("round")["valuable_run_recall"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        ax.plot(
            agg["round"],
            agg["mean"],
            marker="o",
            label=learner
        )

        # Approximate 95% interval across deployment orderings
        se = (
            agg["std"].fillna(0)
            / agg["count"].clip(lower=1).pow(0.5)
        )

        ax.fill_between(
            agg["round"],
            agg["mean"] - 1.96 * se,
            agg["mean"] + 1.96 * se,
            alpha=0.15
        )

    ax.set_xlabel("Deployment round")
    ax.set_ylabel("Cumulative valuable-run recall")

    ax.set_title(
        f"Cumulative recall over deployment rounds "
        f"(stop quantile = {stop_quantile:.2f})"
    )

    # Force integer rounds 1–5
    ax.set_xticks(range(1, max_round + 1))

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "supervisor_03_cumulative_recall_q40_rounds1to5.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 4:
# Scatter plot of early prediction vs final outcome
#
# This tries to load a late-bloomer prediction table if available.
# If your exact file name differs, just edit CANDIDATES below.
# ============================================================

def plot_late_bloomer_scatter():
    CANDIDATES = [
        "03_late_bloomers_predictions.csv",
        "03_late_bloomer_predictions.csv",
        "03_late_bloomers.csv",
        "03_predictions.csv",
    ]

    path = None
    for name in CANDIDATES:
        candidate = TABLE_DIR / name
        if candidate.exists():
            path = candidate
            break

    if path is None:
        print(
            "Skipping scatter plot: no late-bloomer prediction file found.\n"
            "Looked for:\n"
            + "\n".join(str(TABLE_DIR / x) for x in CANDIDATES)
        )
        return

    df = pd.read_csv(path)

    required_cols = {"prediction", "final_v002_purity"}
    if not required_cols.issubset(df.columns):
        print(
            f"Skipping scatter plot: {path.name} does not contain "
            f"the required columns {required_cols}."
        )
        print("Columns found:", df.columns.tolist())
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.scatter(
        df["prediction"],
        df["final_v002_purity"],
        alpha=0.75,
        label="Held-out runs"
    )

    if "late_bloomer" in df.columns:
        late = df[df["late_bloomer"] == True]  # noqa: E712
        if len(late) > 0:
            ax.scatter(
                late["prediction"],
                late["final_v002_purity"],
                marker="x",
                s=80,
                label="Late-bloomer candidates"
            )

    ax.set_xlabel("Early predicted final purity")
    ax.set_ylabel("Observed final purity")
    ax.set_title("Why stopping is difficult: some poor-looking runs finish well")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "supervisor_04_scatter_late_bloomers.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# FIGURE 5:
# Small summary table as an image
# ============================================================

def make_summary_table():
    df = load_csv("05_stop_sweep_summary.csv").copy()

    keep = [
        "stop_quantile",
        "mean_actual_completed_stop_rate",
        "valuable_run_recall_delta_mean",
        "valuable_false_stop_rate_delta_mean",
    ]
    df = df[keep].copy()

    df.columns = [
        "Stop quantile",
        "Actual stop rate",
        "Recall gap\n(completed − oracle)",
        "False-stop gap\n(completed − oracle)"
    ]

    # Round for presentation
    for col in df.columns[1:]:
        df[col] = df[col].round(3)

    save_summary_table_figure(
        df=df,
        filename="supervisor_05_summary_table.png",
        title="Selective-Continuation Trap: summary across stopping thresholds"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    plot_recall_gap_vs_threshold()
    plot_false_stop_gap_vs_threshold()
    plot_cumulative_recall_over_rounds(stop_quantile=0.40)
    plot_late_bloomer_scatter()
    make_summary_table()

    print("\nSaved supervisor-ready visuals to:")
    print(FIG_DIR)
    print("\nGenerated files:")
    print(" - supervisor_01_recall_gap_vs_threshold.png")
    print(" - supervisor_02_false_stop_gap_vs_threshold.png")
    print(" - supervisor_03_cumulative_recall_q40.png")
    print(" - supervisor_04_scatter_late_bloomers.png   (if source file exists)")
    print(" - supervisor_05_summary_table.png")


if __name__ == "__main__":
    main()