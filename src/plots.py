from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR


def plot_prefix_scatter(df: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(df["prediction"], df["final_v002_purity"], alpha=0.75)
    late = df[df.get("late_bloomer", False) == True] if "late_bloomer" in df else pd.DataFrame()  # noqa: E712
    if len(late):
        ax.scatter(late["prediction"], late["final_v002_purity"], marker="x", s=70, label="Late-bloomer candidate")
        ax.legend()
    ax.set_xlabel("Held-out early prediction of final V002 purity")
    ax.set_ylabel("Observed final V002 purity")
    ax.set_title("Early prediction vs final experimental outcome")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def plot_replay(summary: pd.DataFrame, metric: str, filename: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for learner, sub in summary.groupby("learner"):
        agg = sub.groupby("round")[metric].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["round"], agg["mean"], marker="o", label=learner)
        if (agg["count"] > 1).any():
            se = agg["std"].fillna(0) / agg["count"].clip(lower=1).pow(0.5)
            ax.fill_between(agg["round"], agg["mean"] - 1.96 * se, agg["mean"] + 1.96 * se, alpha=0.15)
    ax.set_xlabel("Deployment round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Selective-continuation replay: {metric.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def plot_contrast_sweep(summary, mean_col, lower_col, upper_col, ylabel, filename, title):
    view = summary.sort_values("stop_quantile").copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axhline(0.0, linewidth=1)
    ax.plot(view["stop_quantile"], view[mean_col], marker="o")
    ax.fill_between(view["stop_quantile"], view[lower_col], view[upper_col], alpha=0.18)
    ax.set_xlabel("Stop threshold quantile")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220)
    plt.close(fig)


def plot_cumulative_recall(cumulative: pd.DataFrame, stop_quantile: float, filename: str):
    view = cumulative[cumulative["stop_quantile"] == stop_quantile].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for learner, sub in view.groupby("learner"):
        agg = sub.groupby("round")["valuable_run_recall"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["round"], agg["mean"], marker="o", label=learner)
        se = agg["std"].fillna(0) / agg["count"].clip(lower=1).pow(0.5)
        ax.fill_between(agg["round"], agg["mean"] - 1.96 * se, agg["mean"] + 1.96 * se, alpha=0.15)
    ax.set_xlabel("Deployment round (cumulative)")
    ax.set_ylabel("Cumulative valuable-run recall")
    ax.set_title(f"Cumulative recall at stop quantile = {stop_quantile:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220)
    plt.close(fig)


def plot_recall_gap_vs_actual_stop_rate(paired: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axhline(0.0, linewidth=1)
    ax.scatter(paired["completed_only_stop_rate"], paired["delta_valuable_run_recall"], alpha=0.45)
    ax.set_xlabel("Actual completed-only stop rate")
    ax.set_ylabel("Recall gap: completed-only − oracle")
    ax.set_title("Censoring penalty vs actual stopping aggressiveness")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220)
    plt.close(fig)
