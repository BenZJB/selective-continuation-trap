from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import N_CV_SPLITS, PRIMARY_MIXTURE, PROCESSED_DIR, RANDOM_SEED, TABLE_DIR
from src.metrics import regression_metrics
from src.modeling import fit_predict


def cross_validated_predictions(df: pd.DataFrame, n_splits: int = N_CV_SPLITS) -> pd.DataFrame:
    groups = df["operating_point"].astype(str)
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise RuntimeError("Need at least two operating-point groups for held-out evaluation.")

    cv = GroupKFold(n_splits=n_splits)
    pieces = []
    for fold, (tr, te) in enumerate(cv.split(df, groups=groups)):
        train = df.iloc[tr].copy()
        test = df.iloc[te].copy()
        _, pred, _ = fit_predict(train, test, random_state=RANDOM_SEED + fold)
        test["prediction"] = pred
        test["fold"] = fold
        # Evaluation thresholds are derived only from the training fold.
        test["good_threshold_train"] = train["final_v002_purity"].quantile(0.80)
        train_model, train_pred, _ = fit_predict(train, train, random_state=RANDOM_SEED + fold)
        test["poor_prediction_threshold_train"] = float(np.quantile(train_pred, 0.30))
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def main():
    features = pd.read_csv(PROCESSED_DIR / "prefix_features.csv")
    features = features[(features["mixture"] == PRIMARY_MIXTURE) & features["final_v002_purity"].notna()].copy()

    metric_rows = []
    prediction_rows = []
    for prefix, sub in features.groupby("prefix_fraction"):
        oof = cross_validated_predictions(sub.reset_index(drop=True))
        m = regression_metrics(oof["final_v002_purity"], oof["prediction"])
        metric_rows.append({"prefix_fraction": prefix, "n_runs": len(oof), **m})
        prediction_rows.append(oof)

    metrics = pd.DataFrame(metric_rows).sort_values("prefix_fraction")
    preds = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(TABLE_DIR / "02_prefix_predictability_metrics.csv", index=False)
    preds.to_csv(TABLE_DIR / "02_oof_prefix_predictions.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"\nSaved held-out predictions to {TABLE_DIR / '02_oof_prefix_predictions.csv'}")


if __name__ == "__main__":
    main()
