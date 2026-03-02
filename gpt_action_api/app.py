#!/usr/bin/env python3
"""Standalone FastAPI app for GPT Actions price prediction from BaT listing URLs."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from auction_scraper.scrape_bat_landcruiser_results import fetch, parse_listing

DEFAULT_MODEL_PATH = "auction_scraper/models/landcruiser_price_model.joblib"


class PredictRequest(BaseModel):
    url: str = Field(..., description="Bring a Trailer listing URL")
    model_path: str = Field(DEFAULT_MODEL_PATH, description="Local model artifact path")
    timeout_seconds: float = Field(30.0, ge=5.0, le=120.0)


class PredictionInterval(BaseModel):
    alpha: float
    low_usd: float
    high_usd: float


class PredictResponse(BaseModel):
    url: str
    predicted_price_usd: float
    model_name: str
    feature_columns: list[str]
    features: dict[str, Any]
    interval: Optional[PredictionInterval]
    explanation: list[str]
    extracted_listing: dict[str, Any]


app = FastAPI(
    title="BaT Land Cruiser Price Prediction API",
    version="1.0.0",
    description=(
        "Predicts final sale price for a Bring a Trailer Land Cruiser listing URL using a trained model."
    ),
)


@lru_cache(maxsize=4)
def load_bundle(model_path: str) -> dict[str, Any]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bundle = joblib.load(p)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid model bundle format.")
    return bundle


def build_features(parsed: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
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
        "sale_status": "sold",
        "auction_month": auction_dt.month,
        "auction_quarter": (auction_dt.month - 1) // 3 + 1,
        "auction_year": auction_dt.year,
    }
    row = {c: base.get(c) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def build_explanation(parsed: dict[str, Any], pred: float) -> list[str]:
    out: list[str] = []
    year = parsed.get("year")
    mileage = parsed.get("mileage")
    bids = parsed.get("number_of_bids")

    if year is not None:
        out.append(f"Vehicle year considered: {int(year)}.")
    if mileage is not None:
        out.append(f"Mileage considered: {float(mileage):,.0f} miles.")
    if bids is not None:
        out.append(f"Current/observed bid activity considered: {int(bids)} bids.")

    out.append(f"Predicted central estimate is ${pred:,.0f} based on historical BaT LC100/LC200 outcomes.")
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-from-url", response_model=PredictResponse, operation_id="predictFromUrl")
def predict_from_url(req: PredictRequest) -> PredictResponse:
    if "bringatrailer.com/listing/" not in req.url:
        raise HTTPException(status_code=400, detail="url must be a valid Bring a Trailer listing URL")

    try:
        bundle = load_bundle(req.model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model load failed: {exc}") from exc

    pipeline = bundle["pipeline"]
    feature_columns = bundle.get(
        "feature_columns",
        ["year", "mileage", "number_of_bids", "location", "sale_status"],
    )

    try:
        html = fetch(req.url, timeout=req.timeout_seconds)
        parsed = parse_listing(req.url, html)
        features_df = build_features(parsed, feature_columns)
        pred = float(pipeline.predict(features_df)[0])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"prediction failed: {exc}") from exc

    interval_cfg = bundle.get("prediction_interval", {})
    interval: Optional[PredictionInterval] = None
    try:
        q_hat = float(interval_cfg.get("q_hat", 0.0) or 0.0)
        alpha = float(interval_cfg.get("alpha", 0.1) or 0.1)
        if q_hat > 0:
            interval = PredictionInterval(alpha=alpha, low_usd=pred - q_hat, high_usd=pred + q_hat)
    except Exception:
        interval = None

    return PredictResponse(
        url=req.url,
        predicted_price_usd=pred,
        model_name=str(bundle.get("model_name", "model")),
        feature_columns=list(feature_columns),
        features=features_df.iloc[0].to_dict(),
        interval=interval,
        explanation=build_explanation(parsed, pred),
        extracted_listing={
            "title": parsed.get("title"),
            "year": parsed.get("year"),
            "vin": parsed.get("vin"),
            "mileage": parsed.get("mileage"),
            "location": parsed.get("location"),
            "number_of_bids": parsed.get("number_of_bids"),
            "sale_status": parsed.get("sale_status"),
            "sold_price_usd": parsed.get("sold_price_usd"),
            "highest_bid_usd": parsed.get("highest_bid_usd"),
            "auction_end_datetime_utc": parsed.get("auction_end_datetime_utc"),
        },
    )
