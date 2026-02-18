# Bring a Trailer Land Cruiser 100-Series Live Auction Scraper + API

Target page:
- https://bringatrailer.com/toyota/land-cruiser-100-series/

The script extracts only the **Live Auctions** section and saves results to JSON + CSV.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r auction_scraper/requirements.txt
```

## Run as API (for ChatGPT GPT Actions)

```bash
uvicorn auction_scraper.api:app --host 0.0.0.0 --port 8000
```

Then open:
- `http://localhost:8000/health`
- `http://localhost:8000/live-auctions`
- `http://localhost:8000/openapi.json`

For GPT Actions, deploy this API to a public HTTPS URL, then import `.../openapi.json` in your GPT `Configure -> Actions`.

## One-click Render deploy

This repo includes a Render Blueprint at:
- `render.yaml`

Steps:
1. Push this project to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select your repo; Render will detect `render.yaml`.
4. Deploy, then open:
   - `https://<your-render-domain>/health`
   - `https://<your-render-domain>/openapi.json`
5. In GPT Builder, import the deployed `openapi.json` URL in **Actions**.

## Run

```bash
python auction_scraper/scrape_bat_lc100.py
```

Custom output paths:

```bash
python auction_scraper/scrape_bat_lc100.py \
  --json auction_scraper/live.json \
  --csv auction_scraper/live.csv
```

Also download each live listing HTML page:

```bash
python auction_scraper/scrape_bat_lc100.py \
  --download-pages-dir auction_scraper/pages
```

## Output fields

- `title`
- `url`
- `status` (always `Live`)
- `current_bid`
- `location`

## GPT Action Schema (if not importing openapi.json URL)

```yaml
openapi: 3.1.0
info:
  title: LC100 Auctions API
  version: "1.0.0"
servers:
  - url: https://your-api-domain.com
paths:
  /live-auctions:
    get:
      operationId: getLiveAuctions
      summary: Get active Bring a Trailer Land Cruiser 100-series auctions
      responses:
        "200":
          description: OK
```
