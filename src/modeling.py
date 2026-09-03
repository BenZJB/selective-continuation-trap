from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "run_id",
        "mixture",
        "operating_point",
        "experiment",
        "prefix_fraction",
        "prefix_rows",
        "duration_minutes",
        "final_v002_purity",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def make_regressor(random_state: int = 42):
    """XGBoost baseline with a sklearn fallback.

    XGBoost is the intended baseline. The fallback keeps the project runnable if the
    xgboost wheel is unavailable on a machine.
    """
    try:
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3, random_state=random_state)),
            ]
        )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, random_state: int = 42):
    cols = feature_columns(train)
    model = make_regressor(random_state=random_state)
    model.fit(train[cols], train["final_v002_purity"])
    return model, model.predict(test[cols]), cols
