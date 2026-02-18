#!/usr/bin/env python3
"""Scrape active (live) BaT Land Cruiser 100-Series auctions and save results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

DEFAULT_URL = "https://bringatrailer.com/toyota/land-cruiser-100-series/"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape active (live) Land Cruiser 100-Series auctions from Bring a Trailer"
    )
    p.add_argument("--url", default=DEFAULT_URL, help="BaT series page URL")
    p.add_argument("--json", default="auction_scraper/bat_lc100_live.json", help="Output JSON path")
    p.add_argument("--csv", default="auction_scraper/bat_lc100_live.csv", help="Output CSV path")
    p.add_argument(
        "--download-pages-dir",
        default="",
        help="If set, download each live listing HTML into this directory",
    )
    p.add_argument("--delay", type=float, default=0.8, help="Seconds between detail page downloads")
    p.add_argument("--timeout", type=float, default=30, help="HTTP timeout in seconds")
    return p.parse_args()


def fetch_html(url: str, timeout: float) -> str:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _next_tags(tag: Tag) -> Iterable[Tag]:
    sib = tag.next_sibling
    while sib is not None:
        if isinstance(sib, Tag):
            yield sib
        sib = sib.next_sibling


def extract_live_auctions(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    live_heading = None
    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = " ".join(h.get_text(" ", strip=True).split())
        if re.search(r"\blive auctions\b", txt, flags=re.I):
            live_heading = h
            break

    if live_heading is None:
        raise RuntimeError("Could not find 'Live Auctions' section on the page.")

    auctions: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for node in _next_tags(live_heading):
        if node.name in {"h1", "h2", "h3"}:
            heading_text = node.get_text(" ", strip=True).lower()
            if "auction results" in heading_text or "related stories" in heading_text:
                break

        # BaT listings are usually shown as h3 -> a links in this section.
        for h3 in node.find_all("h3") if node.name != "h3" else [node]:
            a = h3.find("a", href=True)
            if not a:
                continue

            href = a["href"].strip()
            if href.startswith("#"):
                continue

            url = urljoin(base_url, href)
            if url in seen_urls:
                continue

            title = " ".join(a.get_text(" ", strip=True).split())
            if not title:
                continue

            parent_text = " ".join(h3.parent.get_text(" ", strip=True).split()) if h3.parent else ""
            location = ""
            m_loc = re.search(r"\b(USA|Canada|United Kingdom|Australia|Japan|Germany|France)\b", parent_text)
            if m_loc:
                location = m_loc.group(1)

            bid = ""
            m_bid = re.search(r"(?:Bid|Current Bid):\s*([^\n]+)", parent_text, flags=re.I)
            if m_bid:
                bid = m_bid.group(1).strip()

            auctions.append(
                {
                    "title": title,
                    "url": url,
                    "status": "Live",
                    "current_bid": bid,
                    "location": location,
                }
            )
            seen_urls.add(url)

    return auctions


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["title", "url", "status", "current_bid", "location"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:80] or "listing"


def download_listing_pages(rows: list[dict[str, str]], out_dir: Path, timeout: float, delay: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        url = row["url"]
        html = fetch_html(url, timeout=timeout)
        filename = f"{i:03d}_{slugify(row['title'])}.html"
        (out_dir / filename).write_text(html, encoding="utf-8")
        if i < len(rows):
            time.sleep(delay)


def main() -> int:
    args = parse_args()

    html = fetch_html(args.url, timeout=args.timeout)
    auctions = extract_live_auctions(html, base_url=args.url)

    write_json(Path(args.json), auctions)
    write_csv(Path(args.csv), auctions)

    if args.download_pages_dir:
        download_listing_pages(
            auctions,
            out_dir=Path(args.download_pages_dir),
            timeout=args.timeout,
            delay=args.delay,
        )

    print(f"Found {len(auctions)} live auctions")
    print(f"JSON: {args.json}")
    print(f"CSV:  {args.csv}")
    if args.download_pages_dir:
        print(f"Pages: {args.download_pages_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
