# BaT Toyota Land Cruiser: Scraper + Price Prediction App

This project provides:

1. Historical scraper for Bring a Trailer Toyota Land Cruiser listings (auction end dates from **2023-01-01** through current date).
2. Model training + evaluation for final sale price prediction.
3. Simple Streamlit UI where you paste an active auction URL and get a predicted final sale price.

## What gets extracted

Each auction row includes:

- Vehicle info: `title`, `make`, `model`, `year`, `vin`, `mileage`, `location`
- Result info: `sold_price_usd`, `highest_bid_usd` (for reserve-not-met), `sale_status`, `reserve_met`
- Metadata: `auction_end_datetime_utc`, `number_of_bids`, `scraped_at_utc`, `url`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r auction_scraper/requirements.txt
```

## 1) Scrape historical results (2023-current)

```bash
python auction_scraper/scrape_bat_landcruiser_results.py \
  --start-year 2023 \
  --json auction_scraper/data/bat_landcruiser_results_2023_current.json \
  --csv auction_scraper/data/bat_landcruiser_results_2023_current.csv
```

Useful options:

- `--max-pages 120` max index pages to scan
- `--max-listings 0` set >0 for faster sample runs
- `--delay 0.65` polite delay between requests
- `--discovery-method auto|pages|search|wpjson|sitemap|playwright`

For dynamic pages, Playwright discovery can capture additional completed results:

```bash
pip install playwright
playwright install chromium
python auction_scraper/scrape_bat_landcruiser_results.py --discovery-method playwright
```

Playwright mode is results-first:

- Primary source: `https://bringatrailer.com/auctions/results/` search pagination
- Secondary source: series pages (`/toyota/land-cruiser-100-series/`, `/toyota/200-series-land-cruiser/`)
- Uses event-driven `Show More` expansion + network response parsing for better coverage

## 2) Train model and evaluate

```bash
python auction_scraper/train_price_model.py \
  --input auction_scraper/data/bat_landcruiser_results_2023_current.csv \
  --model-out auction_scraper/models/landcruiser_price_model.joblib \
  --metrics-out auction_scraper/models/model_metrics.json \
  --roc-out auction_scraper/models/roc_curve.png
```

Outputs:

- Model artifact: `auction_scraper/models/landcruiser_price_model.joblib`
- Metrics JSON: `auction_scraper/models/model_metrics.json`
- ROC plot: `auction_scraper/models/roc_curve.png`

Metrics include:

- Regression: `MAE`, `RMSE`, `MAPE`, `R²`
- Classification-style proxy on high-value auctions: `accuracy`, `precision`, `recall`, `ROC-AUC`

## 3) Launch the prediction UI

```bash
streamlit run auction_scraper/streamlit_app.py
```

Then in the browser:

1. Keep model path (or set your own).
2. Paste an active BaT listing URL (`https://bringatrailer.com/listing/...`).
3. Click **Predict Sale Price**.

The app extracts listing features from the active URL and predicts final sale price using your trained model.

## Existing API for live auctions

The older FastAPI live-auctions endpoint is still available:

```bash
uvicorn auction_scraper.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `/health`
- `/live-auctions`
