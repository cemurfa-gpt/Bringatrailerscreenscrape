FROM python:3.11-slim

WORKDIR /app
COPY auction_scraper/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY auction_scraper /app/auction_scraper

EXPOSE 8000
CMD ["uvicorn", "auction_scraper.api:app", "--host", "0.0.0.0", "--port", "8000"]

