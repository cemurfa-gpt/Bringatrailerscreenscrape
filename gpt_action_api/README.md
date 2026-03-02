# GPT Action API (Standalone)

This is a separate FastAPI project for ChatGPT Actions so your Streamlit app remains untouched.

## Endpoints

- `GET /health`
- `POST /predict-from-url`

### `POST /predict-from-url` request

```json
{
  "url": "https://bringatrailer.com/listing/...",
  "model_path": "auction_scraper/models/landcruiser_price_model.joblib",
  "timeout_seconds": 30
}
```

### Response

Returns:

- `predicted_price_usd`
- `interval` (`low_usd`, `high_usd`, `alpha`) if model interval exists
- extracted listing fields
- feature values used
- short explanation lines

## Run locally

```bash
cd "/Users/cemurfalioglu/Desktop/Codex Files CU"
source .venv/bin/activate
pip install -r gpt_action_api/requirements.txt
uvicorn gpt_action_api.app:app --host 0.0.0.0 --port 8001
```

Open:

- `http://localhost:8001/health`
- `http://localhost:8001/docs`
- `http://localhost:8001/openapi.json`

## Connect to ChatGPT GPT Actions

1. Deploy this API to a public HTTPS URL.
2. In GPT Builder -> Actions, import `https://<your-domain>/openapi.json`.
3. Add instructions to your GPT to call `predictFromUrl` when user provides a BaT listing URL.
