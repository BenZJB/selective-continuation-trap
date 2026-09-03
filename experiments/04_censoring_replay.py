from __future__ import annotations

import pandas as pd

from config import GOOD_QUANTILE, PRIMARY_MIXTURE, PROCESSED_DIR, REPLAY_SEEDS, STOP_QUANTILES, TABLE_DIR
from src.replay import evaluate_policy, fit_model, make_group_batches
from src.plots import plot_replay


def run_one(df: pd.DataFrame, seed: int, stop_q: float):
    initial_groups, batches = make_group_batches(df, seed)
    initial = df[df["operating_point"].isin(initial_groups)].copy()

    oracle_train = initial.copy()
    censored_train = initial.copy()

    good_threshold = float(initial["final_v002_purity"].quantile(GOOD_QUANTILE))
    stop_threshold = float(initial["final_v002_purity"].quantile(stop_q))

    static_model, static_cols = fit_model(initial, seed + 10000)

    metric_rows, decision_rows = [], []

    for round_id, group_list in enumerate(batches, start=1):
        batch = df[df["operating_point"].isin(group_list)].copy()
        if len(batch) == 0:
            continue

        oracle_model, oracle_cols = fit_model(oracle_train, seed + 100 * round_id + 1)
        censored_model, censored_cols = fit_model(censored_train, seed + 100 * round_id + 2)

        round_decisions = {}

        for learner, model, cols in [
            ("oracle_all_endpoints", oracle_model, oracle_cols),
            ("completed_only", censored_model, censored_cols),
            ("static_initial", static_model, static_cols),
        ]:
            decisions, metrics = evaluate_policy(
                model, cols, batch, stop_threshold, good_threshold,
                learner, seed, stop_q, round_id
            )
            decision_rows.append(decisions)
            metric_rows.append(metrics)
            round_decisions[learner] = decisions

        oracle_train = pd.concat([oracle_train, batch], ignore_index=True)

        censored_decisions = round_decisions["completed_only"]
        continued_ids = censored_decisions.loc[~censored_decisions["stop"], "run_id"]
        observed = batch[batch["run_id"].isin(continued_ids)]
        censored_train = pd.concat([censored_train, observed], ignore_index=True)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    return metrics, decisions


def main():
    features = pd.read_csv(PROCESSED_DIR / "prefix_features.csv")
    df = features[
        (features["mixture"] == PRIMARY_MIXTURE)
        & (features["prefix_fraction"] == 0.20)
    ].copy()
    df = df.dropna(subset=["final_v002_purity"]).reset_index(drop=True)

    all_metrics, all_decisions = [], []
    for seed in REPLAY_SEEDS:
        for stop_q in STOP_QUANTILES:
            m, d = run_one(df, seed=seed, stop_q=stop_q)
            all_metrics.append(m)
            if len(d):
                all_decisions.append(d)

    metrics = pd.concat(all_metrics, ignore_index=True)
    decisions = pd.concat(all_decisions, ignore_index=True)
    metrics.to_csv(TABLE_DIR / "04_censoring_replay_metrics.csv", index=False)
    decisions.to_csv(TABLE_DIR / "04_censoring_replay_decisions.csv", index=False)

    headline = metrics.groupby(["stop_quantile", "learner"])[
        ["valuable_run_recall", "valuable_false_stop_rate", "endpoint_selection_bias", "mae", "spearman"]
    ].mean().reset_index()
    headline.to_csv(TABLE_DIR / "04_censoring_replay_headline.csv", index=False)
    print(headline.to_string(index=False))

    mid_q = float(STOP_QUANTILES[len(STOP_QUANTILES)//2])
    view = metrics[metrics["stop_quantile"] == mid_q].copy()
    plot_replay(view, "valuable_run_recall", "04_replay_valuable_recall.png")
    plot_replay(view, "endpoint_selection_bias", "04_replay_endpoint_selection_bias.png")
    plot_replay(view, "mae", "04_replay_mae.png")

    print("\nInterpretation gate:")
    print("A convincing trap requires completed_only to deteriorate relative to oracle_all_endpoints across rounds/seeds.")
    print("Run `python run_robustness.py` for the fine 50-seed confirmatory sweep.")


if __name__ == "__main__":
    main()
