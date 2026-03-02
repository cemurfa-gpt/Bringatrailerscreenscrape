#!/usr/bin/env python3
"""Train and evaluate a sale price model for BaT Land Cruiser auctions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None

DEFAULT_INPUT = "auction_scraper/data/bat_landcruiser_results_2023_current.csv"
DEFAULT_MODEL = "auction_scraper/models/landcruiser_price_model.joblib"
DEFAULT_METRICS = "auction_scraper/models/model_metrics.json"
DEFAULT_ROC = "auction_scraper/models/roc_curve.png"

TARGET_COLUMN = "sold_price_usd"
BASE_FEATURE_COLUMNS = [
    "year",
    "mileage",
    "number_of_bids",
    "location",
    "sale_status",
    "auction_month",
    "auction_quarter",
    "auction_year",
]
NUMERIC_CANDIDATES = [
    "year",
    "mileage",
    "number_of_bids",
    "auction_month",
    "auction_quarter",
    "auction_year",
]
CATEGORICAL_CANDIDATES = ["location", "sale_status"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Land Cruiser sale price model")
    p.add_argument("--input", default=DEFAULT_INPUT, help="CSV dataset path")
    p.add_argument("--model-out", default=DEFAULT_MODEL, help="Output model artifact")
    p.add_argument("--metrics-out", default=DEFAULT_METRICS, help="Output metrics JSON")
    p.add_argument("--roc-out", default=DEFAULT_ROC, help="Output ROC curve image")
    p.add_argument("--test-size", type=float, default=0.2, help="Chronological test set fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random state")
    p.add_argument(
        "--interval-alpha",
        type=float,
        default=0.1,
        help="Prediction interval miscoverage alpha (0.1 => 90%% interval)",
    )
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def select_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns and df[c].astype(str).str.strip().replace("nan", "").ne("").any()]
    feature_columns = numeric + categorical
    return feature_columns, numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise RuntimeError("No usable features found for modeling.")
    return ColumnTransformer(transformers=transformers)


def build_candidates(random_state: int, pre: ColumnTransformer) -> dict[str, Pipeline]:
    out: dict[str, Pipeline] = {
        "random_forest": Pipeline(
            steps=[
                ("prep", pre),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=18,
                        random_state=random_state,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("prep", pre),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=450,
                        learning_rate=0.035,
                        max_depth=3,
                        subsample=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if XGBRegressor is not None:
        out["xgboost"] = Pipeline(
            steps=[
                ("prep", pre),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=650,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=4,
                    ),
                ),
            ]
        )

    if LGBMRegressor is not None:
        out["lightgbm"] = Pipeline(
            steps=[
                ("prep", pre),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=750,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return out


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["year", "mileage", "number_of_bids", "sold_price_usd", "highest_bid_usd"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    dt = pd.to_datetime(df.get("auction_end_datetime_utc"), errors="coerce", utc=True)
    df["auction_end_dt"] = dt
    df["auction_month"] = dt.dt.month
    df["auction_quarter"] = dt.dt.quarter
    df["auction_year"] = dt.dt.year

    sold_df = df[df["sale_status"].astype(str).str.lower().eq("sold")].copy()
    sold_df = sold_df.dropna(subset=[TARGET_COLUMN])

    if len(sold_df) < 12:
        df["target_price_usd"] = df["sold_price_usd"].fillna(df["highest_bid_usd"])
        sold_df = df.dropna(subset=["target_price_usd"]).copy()
        sold_df[TARGET_COLUMN] = sold_df["target_price_usd"]

    sold_df = sold_df.sort_values(["auction_end_dt", "scraped_at_utc"], na_position="last").reset_index(drop=True)
    return sold_df


def chronological_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_n = max(1, int(round(n * test_size)))
    test_n = min(test_n, n - 1)
    train_df = df.iloc[: n - test_n].copy()
    test_df = df.iloc[n - test_n :].copy()
    return train_df, test_df


def time_series_cv_mae(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float | list[float] | None]:
    n = len(X_train)
    if n < 30:
        return {"folds": 0, "mae_mean": None, "mae_std": None, "mae_per_fold": []}

    n_splits = min(5, max(2, n // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        maes.append(float(mean_absolute_error(y_va, pred)))

    return {
        "folds": len(maes),
        "mae_mean": float(np.mean(maes)) if maes else None,
        "mae_std": float(np.std(maes)) if maes else None,
        "mae_per_fold": maes,
    }


def safe_roc(y_true_cls: np.ndarray, y_score_cls: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    unique = np.unique(y_true_cls)
    if len(unique) < 2:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    auc = float(roc_auc_score(y_true_cls, y_score_cls))
    fpr, tpr, _ = roc_curve(y_true_cls, y_score_cls)
    return auc, fpr, tpr


def conformal_interval_qhat(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
) -> tuple[float, int, str]:
    n = len(X_train)
    calib_n = max(10, int(round(0.2 * n)))
    calib_n = min(calib_n, n - 1)

    proper_X = X_train.iloc[: n - calib_n]
    proper_y = y_train.iloc[: n - calib_n]
    calib_X = X_train.iloc[n - calib_n :]
    calib_y = y_train.iloc[n - calib_n :]

    model.fit(proper_X, proper_y)
    calib_pred = model.predict(calib_X)
    residuals = np.abs(calib_y.to_numpy() - calib_pred)

    q = float(np.quantile(residuals, 1.0 - alpha))
    return q, int(calib_n), "split_conformal"


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    roc_out = Path(args.roc_out)

    df = load_data(input_path)
    if len(df) < 12:
        raise RuntimeError(
            f"Not enough usable auction rows for modeling ({len(df)} rows). Run scraper to collect more data."
        )

    train_df, test_df = chronological_split(df, test_size=args.test_size)

    feature_columns, numeric_features, categorical_features = select_feature_groups(train_df)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(float)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(float)

    pre = build_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
    candidates = build_candidates(random_state=args.random_state, pre=pre)
    model_reports: dict[str, dict[str, float | list[float] | None]] = {}

    for name, model in candidates.items():
        cv_stats = time_series_cv_mae(model, X_train, y_train)
        model_reports[name] = cv_stats

    ranked = [(name, report["mae_mean"]) for name, report in model_reports.items() if report["mae_mean"] is not None]
    if ranked:
        ranked.sort(key=lambda x: float(x[1]))
        best_name = ranked[0][0]
    else:
        best_name = "random_forest"

    pipeline = candidates[best_name]

    # Conformal interval calibration on training history only.
    q_hat, calib_rows, interval_method = conformal_interval_qhat(
        model=candidates[best_name],
        X_train=X_train,
        y_train=y_train,
        alpha=args.interval_alpha,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    lower = y_pred - q_hat
    upper = y_pred + q_hat
    coverage = float(np.mean((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)))
    avg_width = float(np.mean(upper - lower))

    threshold = float(np.median(y_train))
    y_true_cls = (y_test >= threshold).astype(int).to_numpy()
    y_pred_cls = (y_pred >= threshold).astype(int)
    y_score_cls = (y_pred - float(np.min(y_pred))) / (float(np.max(y_pred) - np.min(y_pred)) + 1e-9)

    accuracy = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    auc, fpr, tpr = safe_roc(y_true_cls, y_score_cls)

    ensure_parent(model_out)
    ensure_parent(metrics_out)
    ensure_parent(roc_out)

    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": best_name,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "classification_threshold_median_price": threshold,
            "split_strategy": "chronological",
            "prediction_interval": {
                "method": interval_method,
                "alpha": float(args.interval_alpha),
                "q_hat": float(q_hat),
                "calibration_rows": int(calib_rows),
            },
        },
        model_out,
    )

    metrics = {
        "dataset_rows_used": int(len(df)),
        "split_strategy": "chronological",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target": TARGET_COLUMN,
        "selected_model": best_name,
        "feature_columns_used": feature_columns,
        "candidate_models_cv": model_reports,
        "regression": {
            "mae_usd": float(mae),
            "rmse_usd": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
        },
        "prediction_interval": {
            "method": interval_method,
            "alpha": float(args.interval_alpha),
            "nominal_coverage": float(1.0 - args.interval_alpha),
            "empirical_coverage": coverage,
            "avg_interval_width_usd": avg_width,
            "q_hat_usd": float(q_hat),
            "calibration_rows": int(calib_rows),
        },
        "classification_proxy_high_value": {
            "threshold_median_price_usd": threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc": float(auc),
        },
    }

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(7, 5))
    if np.isnan(auc):
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="AUC unavailable")
    else:
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (High-Value Class Proxy)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_out, dpi=140)

    print(f"Saved model: {model_out}")
    print(f"Saved metrics: {metrics_out}")
    print(f"Saved ROC curve: {roc_out}")
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
