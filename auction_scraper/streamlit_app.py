#!/usr/bin/env python3
"""Simple UI for predicting BaT Land Cruiser auction sale price from an active URL."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from auction_scraper.scrape_bat_landcruiser_results import fetch, parse_listing
except ModuleNotFoundError:
    from scrape_bat_landcruiser_results import fetch, parse_listing

DEFAULT_MODEL = Path("auction_scraper/models/landcruiser_price_model.joblib")


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_reference_stats(csv_path: str) -> dict[str, float]:
    p = Path(csv_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[str, float] = {}
    for col in ["year", "mileage", "number_of_bids", "sold_price_usd"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0:
                out[f"{col}_median"] = float(s.median())
                out[f"{col}_p25"] = float(s.quantile(0.25))
                out[f"{col}_p75"] = float(s.quantile(0.75))
    return out


def get_grouped_feature_importance(bundle: dict, top_n: int = 8) -> list[tuple[str, float]]:
    pipeline = bundle.get("pipeline")
    if pipeline is None:
        return []
    try:
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
    except Exception:
        return []

    if not hasattr(model, "feature_importances_"):
        return []

    try:
        names = prep.get_feature_names_out()
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(names) != len(importances):
            return []
    except Exception:
        return []

    grouped: dict[str, float] = {}
    for name, imp in zip(names, importances):
        if "__" in name:
            _, feat = name.split("__", 1)
        else:
            feat = name
        base = feat.split("_", 1)[0] if feat.startswith("sale_status_") else feat
        grouped[base] = grouped.get(base, 0.0) + float(imp)

    ranked = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def get_local_shap_explanation(bundle: dict, features_df: pd.DataFrame, top_n: int = 8) -> dict:
    try:
        import shap  # type: ignore
    except Exception as exc:
        return {"error": f"SHAP unavailable: {exc}"}

    pipeline = bundle.get("pipeline")
    if pipeline is None:
        return {"error": "Pipeline not found in model bundle."}

    try:
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
    except Exception as exc:
        return {"error": f"Could not access pipeline steps: {exc}"}

    try:
        x_trans = prep.transform(features_df)
        feat_names = prep.get_feature_names_out()
    except Exception as exc:
        return {"error": f"Could not transform features for SHAP: {exc}"}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_trans)
    except Exception as exc:
        return {"error": f"Tree SHAP failed: {exc}"}

    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        row_sv = sv[0]
    elif sv.ndim == 1:
        row_sv = sv
    else:
        return {"error": "Unexpected SHAP value shape."}

    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(base).reshape(-1)[0])
    else:
        base_value = float(base)

    grouped: dict[str, float] = {}
    for name, value in zip(feat_names, row_sv):
        if "__" in name:
            _, feat = name.split("__", 1)
        else:
            feat = name
        base_feat = feat.split("_", 1)[0] if feat.startswith("sale_status_") else feat
        grouped[base_feat] = grouped.get(base_feat, 0.0) + float(value)

    rows = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)
    rows = rows[:top_n]
    positive = [r for r in rows if r[1] > 0][:3]
    negative = [r for r in rows if r[1] < 0][:3]

    return {
        "base_value": base_value,
        "sum_contrib": float(np.sum(row_sv)),
        "top_rows": rows,
        "top_positive": positive,
        "top_negative": negative,
    }


def build_reason_lines(parsed: dict, stats: dict[str, float]) -> list[str]:
    lines: list[str] = []

    year = parsed.get("year")
    mileage = parsed.get("mileage")
    bids = parsed.get("number_of_bids")

    year_med = stats.get("year_median")
    mileage_med = stats.get("mileage_median")
    bids_med = stats.get("number_of_bids_median")

    if year is not None and year_med is not None:
        if float(year) >= year_med:
            lines.append(f"Year `{int(year)}` is newer than dataset median `{year_med:.0f}`, which tends to increase value.")
        else:
            lines.append(f"Year `{int(year)}` is older than dataset median `{year_med:.0f}`, which tends to lower value.")

    if mileage is not None and mileage_med is not None:
        if float(mileage) <= mileage_med:
            lines.append(
                f"Mileage `{float(mileage):,.0f}` is below median `{mileage_med:,.0f}`, which usually supports a higher price."
            )
        else:
            lines.append(
                f"Mileage `{float(mileage):,.0f}` is above median `{mileage_med:,.0f}`, which usually pressures price downward."
            )

    if bids is not None and bids_med is not None:
        if float(bids) >= bids_med:
            lines.append(
                f"Bid count `{int(bids)}` is above median `{bids_med:.0f}`, indicating stronger demand."
            )
        else:
            lines.append(
                f"Bid count `{int(bids)}` is below median `{bids_med:.0f}`, indicating softer demand."
            )

    if not lines:
        lines.append("Prediction is based on learned historical patterns for year, mileage, bids, and auction timing.")

    return lines


def build_features_from_listing(parsed: dict, feature_columns: list[str]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    auction_dt = now
    raw_end = parsed.get("auction_end_datetime_utc")
    if isinstance(raw_end, str) and raw_end.strip():
        try:
            auction_dt = datetime.fromisoformat(raw_end.replace("Z", "+00:00"))
            if auction_dt.tzinfo is None:
                auction_dt = auction_dt.replace(tzinfo=timezone.utc)
        except Exception:
            auction_dt = now
    base = {
        "year": parsed.get("year"),
        "mileage": parsed.get("mileage"),
        "number_of_bids": parsed.get("number_of_bids"),
        "location": parsed.get("location") or "",
        # Model is trained on sold outcomes; keep inference in the same feature space.
        "sale_status": "sold",
        "auction_month": auction_dt.month,
        "auction_quarter": (auction_dt.month - 1) // 3 + 1,
        "auction_year": auction_dt.year,
    }
    row = {c: base.get(c) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def main() -> None:
    st.set_page_config(page_title="BaT Land Cruiser Price Predictor", page_icon="🚗", layout="centered")
    st.title("BaT Land Cruiser Price Predictor")
    st.write("Enter an active Bring a Trailer Land Cruiser auction URL to estimate final sale price.")

    model_path_str = st.text_input("Model path", value=str(DEFAULT_MODEL))
    url = st.text_input("Active auction URL", value="https://bringatrailer.com/listing/")

    if st.button("Predict Sale Price", type="primary"):
        model_path = Path(model_path_str)
        if not model_path.exists():
            st.error(f"Model not found: {model_path}")
            st.stop()

        if "bringatrailer.com/listing/" not in url:
            st.error("Please provide a valid Bring a Trailer listing URL.")
            st.stop()

        with st.spinner("Extracting listing data and generating prediction..."):
            bundle = load_model(model_path)
            pipeline = bundle["pipeline"]
            feature_columns = bundle.get(
                "feature_columns",
                ["year", "mileage", "number_of_bids", "location", "sale_status"],
            )
            interval_cfg = bundle.get("prediction_interval", {})
            stats = load_reference_stats("auction_scraper/data/bat_landcruiser_results_2023_current.csv")
            model_name = bundle.get("model_name", "model")

            html = fetch(url, timeout=30.0)
            parsed = parse_listing(url, html)
            features_df = build_features_from_listing(parsed, feature_columns)
            pred = float(pipeline.predict(features_df)[0])

        st.success("Prediction complete")
        st.metric("Predicted final sale price", f"${pred:,.0f}")
        q_hat = float(interval_cfg.get("q_hat", 0.0) or 0.0)
        if q_hat > 0:
            st.write(
                f"Approx. {(1.0 - float(interval_cfg.get('alpha', 0.1))) * 100:.0f}% prediction interval: "
                f"`${pred - q_hat:,.0f}` to `${pred + q_hat:,.0f}`"
            )

        st.subheader("Why This Prediction")
        st.caption(f"Model: `{model_name}` | Features used: `{', '.join(feature_columns)}`")
        for line in build_reason_lines(parsed, stats):
            st.write(f"- {line}")

        top_importance = get_grouped_feature_importance(bundle, top_n=6)
        if top_importance:
            st.write("Top learned feature influence (global):")
            imp_df = pd.DataFrame(top_importance, columns=["feature", "importance"])
            imp_df["importance"] = imp_df["importance"] / imp_df["importance"].sum()
            st.bar_chart(imp_df.set_index("feature"))

        st.write("Local explanation for this listing (SHAP):")
        local = get_local_shap_explanation(bundle, features_df, top_n=8)
        if local.get("error"):
            st.caption(str(local["error"]))
        else:
            top_rows = local.get("top_rows", [])
            if top_rows:
                loc_df = pd.DataFrame(top_rows, columns=["feature", "contribution_usd"])
                st.dataframe(loc_df, use_container_width=True)
            pos = local.get("top_positive", [])
            neg = local.get("top_negative", [])
            if pos:
                st.write(
                    "Top upward drivers: "
                    + ", ".join([f"`{k}` (+${v:,.0f})" for k, v in pos])
                )
            if neg:
                st.write(
                    "Top downward drivers: "
                    + ", ".join([f"`{k}` (-${abs(v):,.0f})" for k, v in neg])
                )

        st.subheader("Extracted listing features")
        st.json(
            {
                "title": parsed.get("title"),
                "year": parsed.get("year"),
                "vin": parsed.get("vin"),
                "mileage": parsed.get("mileage"),
                "location": parsed.get("location"),
                "number_of_bids": parsed.get("number_of_bids"),
            }
        )

        st.caption(
            "Prediction is based on historical BaT Toyota Land Cruiser auction results (2023-current)."
        )


if __name__ == "__main__":
    main()
