#!/usr/bin/env python3
"""FastAPI wrapper for BaT Land Cruiser 100-series live auctions."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from auction_scraper.scrape_bat_lc100 import DEFAULT_URL, extract_live_auctions, fetch_html


class Auction(BaseModel):
    title: str
    url: str
    status: str
    current_bid: str
    location: str


class LiveAuctionsResponse(BaseModel):
    source: str
    count: int
    auctions: list[Auction]


app = FastAPI(
    title="LC100 Auctions API",
    version="1.0.0",
    description="Returns active Bring a Trailer Land Cruiser 100-series auctions.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/live-auctions", response_model=LiveAuctionsResponse, operation_id="getLiveAuctions")
def live_auctions(
    url: str = Query(DEFAULT_URL, description="BaT series page URL"),
    timeout: float = Query(30.0, ge=5.0, le=120.0, description="HTTP timeout in seconds"),
) -> LiveAuctionsResponse:
    try:
        html = fetch_html(url, timeout=timeout)
        auctions = extract_live_auctions(html, base_url=url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Scrape failed: {exc}") from exc

    return LiveAuctionsResponse(source=url, count=len(auctions), auctions=[Auction(**a) for a in auctions])

