from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "spearman": np.nan}

    rho = spearmanr(y_true[ok], y_pred[ok]).statistic if ok.sum() >= 3 else np.nan
    return {
        "mae": float(mean_absolute_error(y_true[ok], y_pred[ok])),
        "rmse": float(np.sqrt(mean_squared_error(y_true[ok], y_pred[ok]))),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def stopping_metrics(df: pd.DataFrame, good_threshold: float) -> dict[str, float]:
    good = df["true_outcome"] >= good_threshold
    stopped = df["stop"].astype(bool)
    n_good = int(good.sum())
    false_stop = stopped & good

    best_all = float(df["true_outcome"].max()) if len(df) else np.nan
    continued = df.loc[~stopped, "true_outcome"]
    best_observed = float(continued.max()) if len(continued) else np.nan
    regret = best_all - best_observed if np.isfinite(best_observed) else np.nan

    return {
        "n": int(len(df)),
        "n_valuable": n_good,
        "n_false_stops": int(false_stop.sum()),
        "stop_rate": float(stopped.mean()) if len(df) else np.nan,
        "valuable_run_recall": float(1.0 - false_stop.sum() / n_good) if n_good else np.nan,
        "valuable_false_stop_rate": float(false_stop.sum() / n_good) if n_good else np.nan,
        "endpoint_observation_rate": float((~stopped).mean()) if len(df) else np.nan,
        "mean_true_outcome": float(df["true_outcome"].mean()) if len(df) else np.nan,
        "mean_observed_outcome": float(continued.mean()) if len(continued) else np.nan,
        "endpoint_selection_bias": float(continued.mean() - df["true_outcome"].mean()) if len(continued) else np.nan,
        "simple_regret_batch": float(regret) if np.isfinite(regret) else np.nan,
    }


def summarize_decisions(decisions: pd.DataFrame) -> dict[str, float]:
    if decisions.empty:
        return {}

    good_threshold = float(decisions["good_threshold"].iloc[0])
    out = stopping_metrics(decisions, good_threshold)
    out.update(regression_metrics(decisions["true_outcome"], decisions["prediction"]))
    return out


def bootstrap_mean_interval(
    values,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 20260903,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "n": 0,
            "fraction_below_zero": np.nan,
            "fraction_above_zero": np.nan,
        }

    rng = np.random.default_rng(seed)
    if arr.size == 1:
        boot_means = np.repeat(arr[0], n_boot)
    else:
        draw_idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        boot_means = arr[draw_idx].mean(axis=1)

    alpha = (1.0 - level) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "lower": float(np.quantile(boot_means, alpha)),
        "upper": float(np.quantile(boot_means, 1.0 - alpha)),
        "n": int(arr.size),
        "fraction_below_zero": float(np.mean(arr < 0)),
        "fraction_above_zero": float(np.mean(arr > 0)),
    }
