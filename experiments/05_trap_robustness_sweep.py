from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    GOOD_QUANTILE,
    PRIMARY_MIXTURE,
    PROCESSED_DIR,
    ROBUST_BOOTSTRAP_REPS,
    ROBUST_BOOTSTRAP_SEED,
    ROBUST_INTERVAL_LEVEL,
    ROBUST_PREFIX,
    ROBUST_REPLAY_SEEDS,
    ROBUST_STOP_QUANTILES,
    TABLE_DIR,
)
from src.metrics import bootstrap_mean_interval, regression_metrics, stopping_metrics, summarize_decisions
from src.plots import plot_contrast_sweep, plot_cumulative_recall, plot_recall_gap_vs_actual_stop_rate
from src.replay import decision_frame, fit_model, make_group_batches


PRIMARY_LEARNERS = ("oracle_all_endpoints", "completed_only")


def _metric_row(decisions: pd.DataFrame) -> dict:
    good_threshold = float(decisions["good_threshold"].iloc[0])
    out = stopping_metrics(decisions, good_threshold)
    out.update(regression_metrics(decisions["true_outcome"], decisions["prediction"]))
    out.update({
        "seed": int(decisions["seed"].iloc[0]),
        "stop_quantile": float(decisions["stop_quantile"].iloc[0]),
        "learner": str(decisions["learner"].iloc[0]),
        "round": int(decisions["round"].iloc[0]),
    })
    return out


def run_seed(df: pd.DataFrame, seed: int):
    initial_groups, batches = make_group_batches(df, seed)
    initial = df[df["operating_point"].isin(initial_groups)].copy()

    good_threshold = float(initial["final_v002_purity"].quantile(GOOD_QUANTILE))
    stop_thresholds = {
        float(q): float(initial["final_v002_purity"].quantile(q))
        for q in ROBUST_STOP_QUANTILES
    }

    oracle_train = initial.copy()
    completed_train = {float(q): initial.copy() for q in ROBUST_STOP_QUANTILES}

    static_model, static_cols = fit_model(initial, seed + 500_000)

    decision_frames = []
    round_metric_rows = []

    for round_id, group_list in enumerate(batches, start=1):
        batch = df[df["operating_point"].isin(group_list)].copy()
        if batch.empty:
            continue

        oracle_model, oracle_cols = fit_model(oracle_train, seed + 10_000 + round_id)
        oracle_pred = oracle_model.predict(batch[oracle_cols])
        static_pred = static_model.predict(batch[static_cols])

        for q_index, stop_q in enumerate(ROBUST_STOP_QUANTILES):
            stop_q = float(stop_q)
            threshold = stop_thresholds[stop_q]

            oracle_decisions = decision_frame(
                batch, oracle_pred, threshold, good_threshold,
                "oracle_all_endpoints", seed, stop_q, round_id
            )

            static_decisions = decision_frame(
                batch, static_pred, threshold, good_threshold,
                "static_initial", seed, stop_q, round_id
            )

            completed_model, completed_cols = fit_model(
                completed_train[stop_q],
                seed + 100_000 + 1_000 * round_id + q_index,
            )
            completed_pred = completed_model.predict(batch[completed_cols])
            completed_decisions = decision_frame(
                batch, completed_pred, threshold, good_threshold,
                "completed_only", seed, stop_q, round_id
            )

            for decisions in (oracle_decisions, completed_decisions, static_decisions):
                decision_frames.append(decisions)
                round_metric_rows.append(_metric_row(decisions))

            continued_ids = completed_decisions.loc[~completed_decisions["stop"], "run_id"]
            newly_observed = batch[batch["run_id"].isin(continued_ids)]
            completed_train[stop_q] = pd.concat(
                [completed_train[stop_q], newly_observed],
                ignore_index=True,
            )

        oracle_train = pd.concat([oracle_train, batch], ignore_index=True)

    return (
        pd.concat(decision_frames, ignore_index=True),
        pd.DataFrame(round_metric_rows),
    )


def build_seed_level_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, stop_q, learner), group in decisions.groupby(["seed", "stop_quantile", "learner"]):
        metrics = summarize_decisions(group)
        metrics.update({"seed": int(seed), "stop_quantile": float(stop_q), "learner": learner})
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_paired_contrasts(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "valuable_run_recall",
        "valuable_false_stop_rate",
        "stop_rate",
        "endpoint_observation_rate",
        "endpoint_selection_bias",
        "simple_regret_batch",
        "mae",
        "rmse",
        "spearman",
    ]

    base = seed_summary[seed_summary["learner"].isin(PRIMARY_LEARNERS)].copy()
    rows = []

    for (seed, stop_q), group in base.groupby(["seed", "stop_quantile"]):
        indexed = group.set_index("learner")
        if not set(PRIMARY_LEARNERS).issubset(indexed.index):
            continue

        row = {"seed": int(seed), "stop_quantile": float(stop_q)}
        for metric in metrics:
            oracle = float(indexed.loc["oracle_all_endpoints", metric])
            completed = float(indexed.loc["completed_only", metric])
            row[f"oracle_{metric}"] = oracle
            row[f"completed_only_{metric}"] = completed
            row[f"delta_{metric}"] = completed - oracle
        rows.append(row)

    return pd.DataFrame(rows)


def build_stop_sweep_summary(paired: pd.DataFrame) -> pd.DataFrame:
    delta_metrics = [
        "delta_valuable_run_recall",
        "delta_valuable_false_stop_rate",
        "delta_simple_regret_batch",
        "delta_mae",
        "delta_spearman",
        "delta_endpoint_selection_bias",
    ]

    rows = []
    for stop_q, group in paired.groupby("stop_quantile", sort=True):
        row = {
            "stop_quantile": float(stop_q),
            "n_seeds": int(group["seed"].nunique()),
            "mean_actual_completed_stop_rate": float(group["completed_only_stop_rate"].mean()),
            "mean_actual_oracle_stop_rate": float(group["oracle_stop_rate"].mean()),
        }

        for i, metric in enumerate(delta_metrics):
            stats = bootstrap_mean_interval(
                group[metric],
                level=ROBUST_INTERVAL_LEVEL,
                n_boot=ROBUST_BOOTSTRAP_REPS,
                seed=ROBUST_BOOTSTRAP_SEED + i + int(round(stop_q * 1000)),
            )
            stem = metric.replace("delta_", "")
            row[f"{stem}_delta_mean"] = stats["mean"]
            row[f"{stem}_delta_median"] = stats["median"]
            row[f"{stem}_delta_lower"] = stats["lower"]
            row[f"{stem}_delta_upper"] = stats["upper"]
            row[f"{stem}_delta_std"] = stats["std"]
            row[f"{stem}_fraction_delta_below_zero"] = stats["fraction_below_zero"]
            row[f"{stem}_fraction_delta_above_zero"] = stats["fraction_above_zero"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_cumulative_round_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, stop_q, learner), group in decisions.groupby(["seed", "stop_quantile", "learner"]):
        group = group.sort_values("round")
        for round_id in sorted(group["round"].unique()):
            cumulative = group[group["round"] <= round_id]
            metrics = summarize_decisions(cumulative)
            metrics.update({
                "seed": int(seed),
                "stop_quantile": float(stop_q),
                "learner": learner,
                "round": int(round_id),
            })
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_dose_response(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in paired.groupby("seed"):
        group = group.sort_values("stop_quantile")
        x = group["stop_quantile"].to_numpy(dtype=float)

        for metric in (
            "delta_valuable_run_recall",
            "delta_valuable_false_stop_rate",
            "delta_simple_regret_batch",
        ):
            y = group[metric].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            slope = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 2 else np.nan
            rows.append({
                "seed": int(seed),
                "metric": metric,
                "slope_per_unit_quantile": float(slope) if np.isfinite(slope) else np.nan,
                "slope_per_10pct_quantile": float(slope * 0.10) if np.isfinite(slope) else np.nan,
            })
    return pd.DataFrame(rows)


def print_gate_summary(stop_summary: pd.DataFrame, dose: pd.DataFrame):
    cols = [
        "stop_quantile",
        "n_seeds",
        "mean_actual_completed_stop_rate",
        "valuable_run_recall_delta_mean",
        "valuable_run_recall_delta_lower",
        "valuable_run_recall_delta_upper",
        "valuable_run_recall_fraction_delta_below_zero",
        "valuable_false_stop_rate_delta_mean",
        "simple_regret_batch_delta_mean",
    ]
    print("\nPAIRED COMPLETED-ONLY MINUS ORACLE SUMMARY")
    print(stop_summary[cols].to_string(index=False))

    recall_slopes = dose[
        dose["metric"] == "delta_valuable_run_recall"
    ]["slope_per_10pct_quantile"].dropna()

    if len(recall_slopes):
        print("\nDose-response diagnostic:")
        print(
            "Mean change in completed-only minus oracle recall gap "
            f"for each +10 percentage points of threshold: {recall_slopes.mean():.4f}"
        )
        print(
            "Fraction of deployment orderings with a negative recall-gap slope: "
            f"{(recall_slopes < 0).mean():.3f}"
        )

    print("\nINTERPRETATION")
    print(
        "Strong evidence would require recall delta < 0 across most thresholds/orderings, "
        "false-stop delta > 0, a stronger penalty under more aggressive stopping, "
        "and cumulative separation over rounds."
    )
    print(
        "The intervals quantify sensitivity to deployment ordering, not uncertainty "
        "over a new population of physical experiments."
    )


def main():
    features = pd.read_csv(PROCESSED_DIR / "prefix_features.csv")
    df = features[
        (features["mixture"] == PRIMARY_MIXTURE)
        & (features["prefix_fraction"] == ROBUST_PREFIX)
    ].copy()
    df = df.dropna(subset=["final_v002_purity"]).reset_index(drop=True)

    print(
        f"Robustness sweep: {len(ROBUST_REPLAY_SEEDS)} deployment orderings x "
        f"{len(ROBUST_STOP_QUANTILES)} stop thresholds on {len(df)} physical runs."
    )

    decision_parts, round_metric_parts = [], []

    for i, seed in enumerate(ROBUST_REPLAY_SEEDS, start=1):
        decisions, round_metrics = run_seed(df, int(seed))
        decision_parts.append(decisions)
        round_metric_parts.append(round_metrics)

        if i == 1 or i % 5 == 0 or i == len(ROBUST_REPLAY_SEEDS):
            print(f"Completed deployment ordering {i}/{len(ROBUST_REPLAY_SEEDS)}")

    decisions = pd.concat(decision_parts, ignore_index=True)
    round_metrics = pd.concat(round_metric_parts, ignore_index=True)

    seed_summary = build_seed_level_summary(decisions)
    paired = build_paired_contrasts(seed_summary)
    stop_summary = build_stop_sweep_summary(paired)
    cumulative = build_cumulative_round_metrics(decisions)
    dose = build_dose_response(paired)

    decisions.to_csv(TABLE_DIR / "05_robustness_decisions.csv", index=False)
    round_metrics.to_csv(TABLE_DIR / "05_round_metrics.csv", index=False)
    seed_summary.to_csv(TABLE_DIR / "05_seed_level_summary.csv", index=False)
    paired.to_csv(TABLE_DIR / "05_paired_contrasts_by_seed.csv", index=False)
    stop_summary.to_csv(TABLE_DIR / "05_stop_sweep_summary.csv", index=False)
    cumulative.to_csv(TABLE_DIR / "05_cumulative_round_metrics.csv", index=False)
    dose.to_csv(TABLE_DIR / "05_dose_response_by_seed.csv", index=False)

    plot_contrast_sweep(
        stop_summary,
        "valuable_run_recall_delta_mean",
        "valuable_run_recall_delta_lower",
        "valuable_run_recall_delta_upper",
        "Recall gap: completed-only − oracle",
        "05_recall_gap_vs_stop_quantile.png",
        "Selective-censoring penalty across stopping thresholds",
    )
    plot_contrast_sweep(
        stop_summary,
        "valuable_false_stop_rate_delta_mean",
        "valuable_false_stop_rate_delta_lower",
        "valuable_false_stop_rate_delta_upper",
        "False-stop-rate gap: completed-only − oracle",
        "05_false_stop_gap_vs_stop_quantile.png",
        "Valuable false-stop penalty across stopping thresholds",
    )

    if 0.40 in set(cumulative["stop_quantile"]):
        plot_cumulative_recall(cumulative, 0.40, "05_cumulative_recall_q40.png")

    plot_recall_gap_vs_actual_stop_rate(
        paired,
        "05_recall_gap_vs_actual_stop_rate.png",
    )

    print_gate_summary(stop_summary, dose)

    print("\nSaved robustness tables and figures under results/.")


if __name__ == "__main__":
    main()
