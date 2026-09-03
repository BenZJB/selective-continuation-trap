from __future__ import annotations

import math
import numpy as np
import pandas as pd

from config import INITIAL_TRAIN_FRACTION, TARGET_ROUND_RUNS
from src.metrics import regression_metrics, stopping_metrics
from src.modeling import feature_columns, make_regressor


def make_group_batches(
    df: pd.DataFrame,
    seed: int,
    initial_train_fraction: float = INITIAL_TRAIN_FRACTION,
    target_round_runs: int = TARGET_ROUND_RUNS,
):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df["operating_point"].unique()))
    rng.shuffle(groups)

    initial_target = max(12, int(math.ceil(len(df) * initial_train_fraction)))
    initial_groups, remaining_groups = [], []
    count = 0

    for group in groups:
        if count < initial_target:
            initial_groups.append(group)
            count += int((df["operating_point"] == group).sum())
        else:
            remaining_groups.append(group)

    batches = []
    current_groups, current_n = [], 0
    for group in remaining_groups:
        current_groups.append(group)
        current_n += int((df["operating_point"] == group).sum())
        if current_n >= target_round_runs:
            batches.append(current_groups)
            current_groups, current_n = [], 0

    if current_groups:
        batches.append(current_groups)

    return initial_groups, batches


def fit_model(train: pd.DataFrame, seed: int):
    cols = feature_columns(train)
    model = make_regressor(seed)
    model.fit(train[cols], train["final_v002_purity"])
    return model, cols


def decision_frame(
    batch: pd.DataFrame,
    prediction,
    stop_threshold: float,
    good_threshold: float,
    learner: str,
    seed: int,
    stop_quantile: float,
    round_id: int,
) -> pd.DataFrame:
    tmp = batch[["run_id", "operating_point", "final_v002_purity"]].copy()
    tmp["prediction"] = np.asarray(prediction, dtype=float)
    tmp["true_outcome"] = tmp["final_v002_purity"]
    tmp["stop"] = tmp["prediction"] < float(stop_threshold)
    tmp["is_valuable"] = tmp["true_outcome"] >= float(good_threshold)
    tmp["false_stop"] = tmp["stop"] & tmp["is_valuable"]
    tmp["learner"] = learner
    tmp["seed"] = int(seed)
    tmp["stop_quantile"] = float(stop_quantile)
    tmp["round"] = int(round_id)
    tmp["stop_threshold"] = float(stop_threshold)
    tmp["good_threshold"] = float(good_threshold)
    return tmp


def evaluate_policy(
    model,
    cols,
    batch: pd.DataFrame,
    stop_threshold: float,
    good_threshold: float,
    learner: str,
    seed: int,
    stop_quantile: float,
    round_id: int,
):
    pred = model.predict(batch[cols])
    decisions = decision_frame(
        batch=batch,
        prediction=pred,
        stop_threshold=stop_threshold,
        good_threshold=good_threshold,
        learner=learner,
        seed=seed,
        stop_quantile=stop_quantile,
        round_id=round_id,
    )
    metrics = stopping_metrics(decisions, good_threshold)
    metrics.update(regression_metrics(decisions["true_outcome"], decisions["prediction"]))
    metrics.update(
        {
            "learner": learner,
            "seed": seed,
            "stop_quantile": stop_quantile,
            "round": round_id,
        }
    )
    return decisions, metrics
