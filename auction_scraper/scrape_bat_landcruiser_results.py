#!/usr/bin/env python3
"""Scrape Bring a Trailer Toyota Land Cruiser auction results and metadata."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from requests import Response

BASE_URL = "https://bringatrailer.com/toyota/land-cruiser/"
DEFAULT_SITEMAP_CANDIDATES = [
    "https://bringatrailer.com/sitemap.xml",
    "https://bringatrailer.com/sitemap_index.xml",
    "https://bringatrailer.com/wp-sitemap.xml",
]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PLAYWRIGHT_LOAD_MORE_SELECTORS = [
    "button:has-text('Load More')",
    "button:has-text('Show More')",
    "button:has-text('More Results')",
    "button:has-text('Load Results')",
    "a:has-text('Load More')",
    "a:has-text('Show More')",
    "a:has-text('More Results')",
    "a:has-text('Load Results')",
    "[data-testid='load-more']",
]


@dataclass
class ScrapeConfig:
    start_year: int
    end_date_utc: datetime
    timeout: float
    delay: float
    max_pages: int
    max_listings: int
    discovery_method: str
    sitemap_index_url: str
    wpjson_max_pages: int
    playwright_max_clicks: int
    detail_fetch: str
    results_search_terms: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape BaT Toyota Land Cruiser auction results")
    p.add_argument("--base-url", default=BASE_URL, help="Base BaT model URL")
    p.add_argument(
        "--base-urls",
        nargs="+",
        default=[],
        help="Optional list of BaT model/series URLs to crawl in one run",
    )
    p.add_argument("--start-year", type=int, default=2023, help="Earliest auction end year to include")
    p.add_argument(
        "--json",
        default="auction_scraper/data/bat_landcruiser_results_2023_current.json",
        help="Output JSON file",
    )
    p.add_argument(
        "--csv",
        default="auction_scraper/data/bat_landcruiser_results_2023_current.csv",
        help="Output CSV file",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    p.add_argument("--delay", type=float, default=0.65, help="Delay between requests")
    p.add_argument("--max-pages", type=int, default=120, help="Maximum index pages to scan")
    p.add_argument("--max-listings", type=int, default=0, help="Optional cap on listing pages (0 = no cap)")
    p.add_argument(
        "--discovery-method",
        choices=["auto", "sitemap", "pages", "wpjson", "search", "playwright"],
        default="auto",
        help="How to discover listing URLs",
    )
    p.add_argument(
        "--sitemap-index-url",
        default="",
        help="Optional sitemap index URL override for URL discovery",
    )
    p.add_argument(
        "--wpjson-max-pages",
        type=int,
        default=60,
        help="Maximum pages to scan when using wp-json discovery",
    )
    p.add_argument(
        "--playwright-max-clicks",
        type=int,
        default=40,
        help="Max number of 'load more' clicks when using playwright discovery",
    )
    p.add_argument(
        "--detail-fetch",
        choices=["auto", "requests", "playwright"],
        default="auto",
        help="How to fetch individual listing detail pages for parsing",
    )
    p.add_argument(
        "--results-search-terms",
        nargs="+",
        default=[],
        help="Optional search terms for BaT /auctions/results discovery in playwright mode",
    )
    p.add_argument(
        "--strict-end-date",
        action="store_true",
        help="If set, drop rows that do not have parseable auction_end_datetime_utc",
    )
    return p.parse_args()


def fetch(url: str, timeout: float) -> str:
    html, _ = fetch_with_final_url(url, timeout)
    return html


def fetch_response(url: str, timeout: float) -> Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_with_final_url(url: str, timeout: float) -> tuple[str, str]:
    resp = fetch_response(url, timeout)
    return resp.text, resp.url


def slug_to_title(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+$", "", slug)
    return " ".join(part.capitalize() for part in slug.split("-") if part)


def expand_playwright_results(page: Any, max_clicks: int, timeout_ms: int) -> None:
    for _ in range(max(0, max_clicks)):
        clicked = False
        for sel in PLAYWRIGHT_LOAD_MORE_SELECTORS:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    loc.click(timeout=min(2500, timeout_ms))
                    try:
                        page.wait_for_load_state("networkidle", timeout=min(5000, timeout_ms))
                    except Exception:
                        page.wait_for_timeout(900)
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            break


def to_int(value: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def to_float_miles(value: str) -> float | None:
    cleaned = value.lower().replace(",", "").strip()
    try:
        if cleaned.endswith("k"):
            return float(cleaned[:-1]) * 1000.0
        return float(cleaned)
    except ValueError:
        return None


def parse_money(text: str) -> int | None:
    m = re.search(r"\$\s*([\d,]+)", text)
    return to_int(m.group(1)) if m else None


def parse_datetime(value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = dtparser.parse(value, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.year < 2000:
            return ""
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def parse_unix_timestamp(value: str | int | None) -> str:
    raw = str(value or "").strip()
    if not raw.isdigit():
        return ""
    try:
        ts = int(raw)
        if ts > 10_000_000_000:
            ts = ts // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.year < 2000:
            return ""
        return dt.isoformat()
    except Exception:
        return ""


def parse_mileage_from_title(title: str) -> float | None:
    patterns = [
        r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k)?[-\s]*mile",
        r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k)\b",
    ]
    for pat in patterns:
        m = re.search(pat, title, flags=re.I)
        if not m:
            continue
        value = str(m.group(1)).replace(",", "")
        suffix_k = (m.group(2) or "").lower() == "k"
        try:
            miles = float(value)
            if suffix_k:
                miles *= 1000.0
            return miles
        except ValueError:
            continue
    return None


def is_lc100_or_lc200_listing(title: str, url: str) -> bool:
    t = (title or "").lower()
    u = (url or "").lower()
    keep_patterns = [
        "land cruiser 100",
        "land cruiser 200",
        "100 series",
        "200 series",
        "uzj100",
        "hdj100",
        "hj100",
        "urj200",
        "vdj200",
        "toyota land cruiser",
        "lexus lx470",
        "lexus lx570",
    ]
    block_patterns = [
        "fj40",
        "fj45",
        "fj55",
        "fj60",
        "fj62",
        "fj70",
        "fj80",
        "fj cruiser",
        "prado",
        "bj40",
        "hj45",
    ]
    combined = f"{t} {u}"
    if any(p in combined for p in block_patterns):
        return False
    if any(p in combined for p in keep_patterns):
        return True
    # If explicitly says Land Cruiser with year >= 1998, it's likely 100/200 generation.
    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", combined)
    if "land cruiser" in combined and m_year:
        year = int(m_year.group(1))
        return year >= 1998
    return False


def extract_rows_from_series_cards_html(html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if "/listing/" not in href:
            continue
        url = urljoin(base_url, href).split("?")[0].rstrip("/")
        if not re.search(r"/listing/[^/]+$", url):
            continue
        if url in seen:
            continue

        card_node = a
        hops = 0
        while getattr(card_node, "parent", None) is not None and hops < 7:
            txt = card_node.get_text(" ", strip=True)
            if txt and ("Sold for" in txt or "Bid to" in txt or "USD $" in txt):
                break
            card_node = card_node.parent
            hops += 1

        card_text = card_node.get_text(" ", strip=True) if card_node else ""
        title = " ".join(a.get_text(" ", strip=True).split())
        if not title:
            h = card_node.find(["h3", "h2"]) if card_node else None
            if h:
                title = " ".join(h.get_text(" ", strip=True).split())
        if not title:
            continue

        if not is_lc100_or_lc200_listing(title, url):
            continue

        sold_match = re.search(r"Sold\s+for\s+(?:USD\s*)?\$?([\d,]+)", card_text, flags=re.I)
        bid_match = re.search(r"Bid\s+to\s+(?:USD\s*)?\$?([\d,]+)", card_text, flags=re.I)
        bids_match = re.search(r"\b(\d{1,4})\s+bids?\b", card_text, flags=re.I)
        date_match = re.search(
            r"\bon\s+([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4})",
            card_text,
            flags=re.I,
        )

        sale_status = "unknown"
        sold_price = None
        highest_bid = None
        reserve_met = None
        if sold_match:
            sold_price = to_int(sold_match.group(1))
            sale_status = "sold"
            reserve_met = True
        elif bid_match:
            highest_bid = to_int(bid_match.group(1))
            sale_status = "reserve_not_met"
            reserve_met = False
        else:
            # Skip cards that do not clearly indicate completed auction result.
            continue

        year = None
        m_year = re.search(r"\b(19\d{2}|20\d{2})\b", title)
        if m_year:
            year = int(m_year.group(1))

        auction_end = parse_datetime(date_match.group(1)) if date_match else ""

        row = {
            "url": url,
            "title": title,
            "make": "Toyota" if re.search(r"\bToyota\b", title, flags=re.I) else "",
            "model": "Land Cruiser" if re.search(r"\bLand\s+Cruiser\b", title, flags=re.I) else "",
            "year": year,
            "vin": "",
            "mileage": parse_mileage_from_title(title),
            "location": "",
            "sale_status": sale_status,
            "reserve_met": reserve_met,
            "sold_price_usd": sold_price,
            "highest_bid_usd": highest_bid,
            "auction_end_datetime_utc": auction_end,
            "number_of_bids": int(bids_match.group(1)) if bids_match else None,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        seen.add(url)

    return rows


def discover_rows_with_playwright(config: ScrapeConfig, base_url: str) -> list[dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    aggregated: dict[str, dict[str, Any]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        seen_page_sigs: set[str] = set()
        page_base = base_url if base_url.endswith("/") else (base_url + "/")

        for pageno in range(1, max(2, config.max_pages + 1)):
            page_url = page_base if pageno == 1 else urljoin(page_base, f"page/{pageno}/")
            try:
                page.goto(page_url, wait_until="networkidle", timeout=int(config.timeout * 1000))
            except Exception:
                break

            for _ in range(max(0, getattr(config, "playwright_max_clicks", 40))):
                clicked = False
                for sel in [
                    "button:has-text('Load More')",
                    "button:has-text('Show More')",
                    "button:has-text('More Results')",
                    "a:has-text('Load More')",
                    "a:has-text('Show More')",
                    "a:has-text('More Results')",
                    "[data-testid='load-more']",
                ]:
                    try:
                        loc = page.locator(sel).first
                        if loc.count() > 0 and loc.is_visible():
                            loc.click(timeout=2000)
                            page.wait_for_timeout(900)
                            clicked = True
                            break
                    except Exception:
                        continue
                if not clicked:
                    break

            html = page.content()
            rows = extract_rows_from_series_cards_html(html, page_url)
            sig = hashlib.sha1("||".join(sorted(r["url"] for r in rows)).encode("utf-8")).hexdigest() if rows else ""
            if sig and sig in seen_page_sigs:
                break
            if sig:
                seen_page_sigs.add(sig)

            before = len(aggregated)
            for row in rows:
                aggregated[row["url"]] = row
            if len(aggregated) == before and pageno > 3:
                break
            if config.max_listings > 0 and len(aggregated) >= config.max_listings:
                break

        context.close()
        browser.close()

    out = list(aggregated.values())
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def discover_rows_from_results_search_playwright(config: ScrapeConfig) -> list[dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    terms = config.results_search_terms or [
        "land cruiser 100 series",
        "land cruiser 200 series",
        "UZJ100",
        "URJ200",
    ]

    aggregated: dict[str, dict[str, Any]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        network_hints: dict[str, dict[str, Any]] = {}

        def on_response(resp: Any) -> None:
            try:
                ctype = str(resp.headers.get("content-type") or "").lower()
                if "json" not in ctype and "html" not in ctype and "javascript" not in ctype:
                    return
                payload = resp.text()
                _, hints = extract_urls_and_hints_from_payload(payload, "https://bringatrailer.com/")
                for u, h in hints.items():
                    prev = network_hints.get(u.rstrip("/"), {})
                    prev.update(h)
                    network_hints[u.rstrip("/")] = prev
            except Exception:
                return

        page.on("response", on_response)

        for term in terms:
            q = requests.utils.quote(term, safe="")
            seen_sigs: set[str] = set()

            for pageno in range(1, max(2, config.max_pages + 1)):
                url = (
                    f"https://bringatrailer.com/auctions/results/?search={q}"
                    if pageno == 1
                    else f"https://bringatrailer.com/auctions/results/page/{pageno}/?search={q}"
                )
                try:
                    page.goto(url, wait_until="networkidle", timeout=int(config.timeout * 1000))
                except Exception:
                    break

                expand_playwright_results(
                    page,
                    max_clicks=config.playwright_max_clicks,
                    timeout_ms=int(config.timeout * 1000),
                )
                html = page.content()
                rows = extract_rows_from_series_cards_html(html, url)
                # Enrich card rows with network-derived hints and add network-only rows.
                for row in rows:
                    h = network_hints.get(row["url"].rstrip("/"), {})
                    if h.get("sold_price_usd") is not None and row.get("sold_price_usd") is None:
                        row["sold_price_usd"] = h.get("sold_price_usd")
                        row["sale_status"] = "sold"
                        row["reserve_met"] = True
                    if h.get("highest_bid_usd") is not None and row.get("highest_bid_usd") is None:
                        row["highest_bid_usd"] = h.get("highest_bid_usd")
                    if h.get("number_of_bids") is not None and row.get("number_of_bids") is None:
                        row["number_of_bids"] = h.get("number_of_bids")
                    if h.get("auction_end_datetime_utc") and not row.get("auction_end_datetime_utc"):
                        row["auction_end_datetime_utc"] = h.get("auction_end_datetime_utc")

                row_by_url = {r["url"].rstrip("/"): r for r in rows}
                for u, h in network_hints.items():
                    if u in row_by_url:
                        continue
                    title = slug_to_title(u)
                    if not is_lc100_or_lc200_listing(title, u):
                        continue
                    if not (h.get("sold_price_usd") or h.get("highest_bid_usd")):
                        continue
                    year = None
                    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", title)
                    if m_year:
                        year = int(m_year.group(1))
                    rows.append(
                        {
                            "url": u,
                            "title": title,
                            "make": "Toyota" if "toyota" in title.lower() else "",
                            "model": "Land Cruiser" if "land cruiser" in title.lower() else "",
                            "year": year,
                            "vin": "",
                            "mileage": parse_mileage_from_title(title),
                            "location": "",
                            "sale_status": h.get("sale_status") or ("sold" if h.get("sold_price_usd") else "reserve_not_met"),
                            "reserve_met": True if h.get("sold_price_usd") else False,
                            "sold_price_usd": h.get("sold_price_usd"),
                            "highest_bid_usd": h.get("highest_bid_usd"),
                            "auction_end_datetime_utc": h.get("auction_end_datetime_utc") or "",
                            "number_of_bids": h.get("number_of_bids"),
                            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                sig = hashlib.sha1("||".join(sorted(r["url"] for r in rows)).encode("utf-8")).hexdigest() if rows else ""
                if sig and sig in seen_sigs:
                    break
                if sig:
                    seen_sigs.add(sig)

                before = len(aggregated)
                for row in rows:
                    aggregated[row["url"]] = row
                if len(aggregated) == before and pageno > 3:
                    break
                if config.max_listings > 0 and len(aggregated) >= config.max_listings:
                    break

            if config.max_listings > 0 and len(aggregated) >= config.max_listings:
                break

        context.close()
        browser.close()

    out = list(aggregated.values())
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def extract_listing_urls(index_html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/listing/" not in href:
            continue
        if href.startswith("#"):
            continue
        full = urljoin(page_url, href)
        full = full.split("?")[0]
        if re.search(r"/listing/[^/]+/?$", full):
            urls.add(full)

    # Some BaT pages embed additional listing URLs inside JSON/script payloads.
    script_matches = re.findall(
        r"(https?:\\/\\/bringatrailer\\.com\\/listing\\/[^\"'\\s<]+|https?://bringatrailer\\.com/listing/[^\"'\\s<]+|\\/listing\\/[^\"'\\s<]+|/listing/[^\"'\\s<]+)",
        index_html,
        flags=re.I,
    )
    for raw in script_matches:
        raw = raw.strip().replace("\\/", "/")
        full = urljoin(page_url, raw).split("?")[0]
        if re.search(r"/listing/[^/]+/?$", full):
            urls.add(full)
    return sorted(urls)


def extract_urls_and_hints_from_payload(payload: str, base_url: str) -> tuple[set[str], dict[str, dict[str, Any]]]:
    urls: set[str] = set()
    hints: dict[str, dict[str, Any]] = {}
    if not payload:
        return urls, hints

    url_matches = list(
        re.finditer(
            r"(https?:\\/\\/bringatrailer\\.com\\/listing\\/[^\"'\\s<]+|https?://bringatrailer\\.com/listing/[^\"'\\s<]+|\\/listing\\/[^\"'\\s<]+|/listing/[^\"'\\s<]+)",
            payload,
            flags=re.I,
        )
    )
    for m in url_matches:
        raw = m.group(1).strip().replace("\\/", "/")
        full = urljoin(base_url, raw).split("?")[0].rstrip("/")
        if not re.search(r"/listing/[^/]+$", full):
            continue
        urls.add(full)

        # Parse hint fields from nearby payload text around this URL match.
        start = max(0, m.start() - 500)
        end = min(len(payload), m.end() + 500)
        window = payload[start:end]
        hint = hints.get(full, {})

        sold = re.search(r"Sold\s+for\s+\$([\d,]+)", window, flags=re.I)
        bid_to = re.search(r"Bid\s+to\s+\$([\d,]+)", window, flags=re.I)
        bids = re.search(r"\b(\d{1,4})\s+bids?\b", window, flags=re.I)
        if sold:
            hint["sold_price_usd"] = to_int(sold.group(1))
            hint["sale_status"] = "sold"
        elif bid_to:
            hint["highest_bid_usd"] = to_int(bid_to.group(1))
            hint["sale_status"] = "reserve_not_met"
        if bids:
            hint["number_of_bids"] = int(bids.group(1))

        # Common timestamp keys seen in HTML fragments/JSON payloads.
        for pat in [
            r'data-featured_listing_ends\\?"?\s*[:=]\s*"?(\d{10,13})',
            r'data-featured-listing-ends\\?"?\s*[:=]\s*"?(\d{10,13})',
            r'"endedAt"\s*:\s*"([^"]+)"',
            r'"closed_at"\s*:\s*"([^"]+)"',
            r'"ends_at"\s*:\s*"([^"]+)"',
            r'"endDate"\s*:\s*"([^"]+)"',
        ]:
            dm = re.search(pat, window, flags=re.I)
            if not dm:
                continue
            raw_date = dm.group(1)
            parsed = parse_unix_timestamp(raw_date) or parse_datetime(raw_date)
            if parsed:
                hint["auction_end_datetime_utc"] = parsed
                break

        hints[full] = hint

    return urls, hints


def extract_next_page(index_html: str, page_url: str) -> str:
    soup = BeautifulSoup(index_html, "html.parser")
    next_link = soup.find("a", rel=lambda v: v and "next" in v)
    if next_link and next_link.get("href"):
        return urljoin(page_url, next_link["href"])

    candidates = soup.find_all("a", href=True)
    for a in candidates:
        txt = a.get_text(" ", strip=True).lower()
        if txt in {"next", "next page"}:
            return urljoin(page_url, a["href"])
    return ""


def extract_canonical(index_html: str, page_url: str) -> str:
    soup = BeautifulSoup(index_html, "html.parser")
    tag = soup.find("link", rel="canonical")
    if tag and tag.get("href"):
        return urljoin(page_url, str(tag.get("href")))
    return ""


def parse_sitemap_xml(xml_text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for elem in root.iter():
        if not elem.tag.endswith("url"):
            continue
        loc = ""
        lastmod = ""
        for child in list(elem):
            if child.tag.endswith("loc"):
                loc = (child.text or "").strip()
            elif child.tag.endswith("lastmod"):
                lastmod = (child.text or "").strip()
        if loc:
            out.append((loc, lastmod))

    if out:
        return out

    for elem in root.iter():
        if not elem.tag.endswith("sitemap"):
            continue
        loc = ""
        lastmod = ""
        for child in list(elem):
            if child.tag.endswith("loc"):
                loc = (child.text or "").strip()
            elif child.tag.endswith("lastmod"):
                lastmod = (child.text or "").strip()
        if loc:
            out.append((loc, lastmod))
    return out


def parse_sitemap_bytes(content: bytes) -> list[tuple[str, str]]:
    payload = content
    if len(content) >= 2 and content[0] == 0x1F and content[1] == 0x8B:
        try:
            payload = gzip.decompress(content)
        except Exception:
            payload = content
    try:
        text = payload.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return parse_sitemap_xml(text)


def discover_urls_from_sitemap(config: ScrapeConfig) -> list[str]:
    urls: set[str] = set()
    index_candidates = [config.sitemap_index_url] if config.sitemap_index_url else DEFAULT_SITEMAP_CANDIDATES
    entries: list[tuple[str, str]] = []
    chosen_index = ""
    for candidate in index_candidates:
        if not candidate:
            continue
        try:
            index_resp = requests.get(
                candidate,
                headers={"User-Agent": USER_AGENT},
                timeout=config.timeout,
            )
            index_resp.raise_for_status()
            parsed_entries = parse_sitemap_bytes(index_resp.content)
            if parsed_entries:
                entries = parsed_entries
                chosen_index = candidate
                break
        except Exception:
            continue

    if not entries:
        raise RuntimeError(
            "Could not load any sitemap index. Tried: " + ", ".join(index_candidates)
        )
    print(f"Sitemap index used: {chosen_index}")
    listing_sitemap_urls = [loc for loc, _ in entries if "listing" in loc and "sitemap" in loc]
    all_sitemap_urls = [loc for loc, _ in entries if "sitemap" in loc]
    # Prefer listing sitemap URLs first, but still scan all sitemap files to avoid missing URLs
    # when the index naming convention differs.
    sitemap_urls = list(dict.fromkeys(listing_sitemap_urls + all_sitemap_urls))
    print(
        "Sitemap candidates:",
        len(sitemap_urls),
        f"(listing-specific: {len(listing_sitemap_urls)}, total: {len(all_sitemap_urls)})",
    )

    lc_url_patterns = [
        "land-cruiser",
        "toyota-fj",
        "fj4",
        "fj5",
        "fj6",
        "fj7",
        "fj8",
        "bj4",
        "bj6",
        "hj4",
        "hj6",
        "fzj",
        "hzj",
        "hdj",
        "vdj",
        "uzj",
        "prado",
        "toyota-land",
    ]

    for idx, sitemap_url in enumerate(sitemap_urls, start=1):
        try:
            resp = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=config.timeout)
            resp.raise_for_status()
        except Exception:
            continue
        before_count = len(urls)
        for loc, lastmod in parse_sitemap_bytes(resp.content):
            if "/listing/" not in loc:
                continue
            low = loc.lower()
            if "bringatrailer.com/listing/" not in low:
                continue
            # Keep Toyota Land Cruiser-like listing slugs, including common chassis-code patterns.
            if "toyota" not in low:
                continue
            if not any(p in low for p in lc_url_patterns):
                continue
            urls.add(loc.split("?")[0])
        after_count = len(urls)
        if after_count > before_count:
            print(f"Sitemap {idx}/{len(sitemap_urls)} added {after_count - before_count} listing URLs")

        if config.max_listings > 0 and len(urls) >= config.max_listings:
            break
        time.sleep(config.delay)

    out = sorted(urls)
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def discover_urls_from_pages(config: ScrapeConfig, base_url: str) -> list[str]:
    page_url = base_url
    page_num = 1
    seen_pages: set[str] = set()
    seen_page_signatures: set[str] = set()
    listing_urls: list[str] = []
    listing_seen: set[str] = set()

    for _ in range(config.max_pages):
        if not page_url or page_url in seen_pages:
            break

        html, final_url = fetch_with_final_url(page_url, timeout=config.timeout)
        # Track requested/final URL only; canonical can point to the base URL for
        # paginated pages and would incorrectly collapse traversal.
        seen_pages.add(page_url)
        seen_pages.add(final_url)

        urls = extract_listing_urls(html, page_url)
        page_sig = hashlib.sha1("||".join(urls).encode("utf-8")).hexdigest() if urls else ""
        if page_sig and page_sig in seen_page_signatures:
            break
        if page_sig:
            seen_page_signatures.add(page_sig)

        for url in urls:
            if url not in listing_seen:
                listing_seen.add(url)
                listing_urls.append(url)
                if config.max_listings > 0 and len(listing_urls) >= config.max_listings:
                    break

        if config.max_listings > 0 and len(listing_urls) >= config.max_listings:
            break

        next_url = extract_next_page(html, final_url)
        if not next_url:
            next_url = urljoin(base_url if base_url.endswith("/") else (base_url + "/"), f"page/{page_num + 1}/")
        if next_url in {page_url, final_url}:
            break
        if not next_url:
            break
        page_url = next_url
        page_num += 1
        time.sleep(config.delay)

    return listing_urls


def discover_urls_from_wp_json(config: ScrapeConfig, base_url: str) -> list[str]:
    root = "https://bringatrailer.com"
    urls: set[str] = set()
    keywords = [
        "toyota land cruiser",
        "land cruiser 100",
        "land cruiser 200",
        "lexus lx470",
        "lexus lx570",
    ]

    endpoint_templates = [
        root + "/wp-json/wp/v2/search?search={q}&per_page=100&page={page}",
        root + "/wp-json/wp/v2/posts?search={q}&per_page=100&page={page}",
        root + "/wp-json/wp/v2/listing?search={q}&per_page=100&page={page}",
    ]

    def collect_from_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        candidates: list[str] = []
        for k in ["url", "link"]:
            v = item.get(k)
            if isinstance(v, str):
                candidates.append(v)
        title_obj = item.get("title")
        if isinstance(title_obj, dict):
            rendered = title_obj.get("rendered")
            if isinstance(rendered, str):
                for m in re.findall(r"https?://bringatrailer\\.com/listing/[^\"'\\s<]+", rendered, flags=re.I):
                    candidates.append(m)
        for v in item.values():
            if isinstance(v, str) and "/listing/" in v and "bringatrailer.com" in v:
                candidates.append(v)
        for raw in candidates:
            low = raw.lower()
            if "bringatrailer.com/listing/" not in low:
                continue
            if "land-cruiser" not in low and "lx470" not in low and "lx570" not in low:
                continue
            urls.add(raw.split("?")[0].rstrip("/"))

    for q in keywords:
        q_enc = requests.utils.quote(q, safe="")
        for template in endpoint_templates:
            empty_streak = 0
            for page in range(1, config.wpjson_max_pages + 1):
                endpoint = template.format(q=q_enc, page=page)
                try:
                    resp = requests.get(endpoint, headers={"User-Agent": USER_AGENT}, timeout=config.timeout)
                except Exception:
                    break
                if resp.status_code in {400, 404}:
                    break
                if resp.status_code == 429:
                    time.sleep(config.delay * 3)
                    continue
                if resp.status_code >= 500:
                    break
                try:
                    payload = resp.json()
                except Exception:
                    break
                if not isinstance(payload, list) or not payload:
                    empty_streak += 1
                    if empty_streak >= 2:
                        break
                    continue
                before = len(urls)
                for item in payload:
                    collect_from_item(item)
                if len(urls) == before:
                    empty_streak += 1
                    if empty_streak >= 3:
                        break
                else:
                    empty_streak = 0
                if config.max_listings > 0 and len(urls) >= config.max_listings:
                    out = sorted(urls)
                    return out[: config.max_listings]
                time.sleep(config.delay)
    out = sorted(urls)
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def discover_urls_from_search(config: ScrapeConfig, base_url: str) -> list[str]:
    urls: set[str] = set()
    query_terms = [
        "toyota land cruiser",
        "land cruiser 100 series",
        "land cruiser 200 series",
        "lexus lx470",
        "lexus lx570",
    ]
    for term in query_terms:
        q = requests.utils.quote(term, safe="")
        for page in range(1, config.max_pages + 1):
            if page == 1:
                search_url = f"https://bringatrailer.com/auctions/?search={q}"
            else:
                search_url = f"https://bringatrailer.com/auctions/page/{page}/?search={q}"
            try:
                html, final_url = fetch_with_final_url(search_url, timeout=config.timeout)
            except Exception:
                break
            found = extract_listing_urls(html, final_url)
            before = len(urls)
            for u in found:
                low = u.lower()
                if "bringatrailer.com/listing/" not in low:
                    continue
                if "toyota" not in low and "land-cruiser" not in low and "lx470" not in low and "lx570" not in low:
                    continue
                urls.add(u.rstrip("/"))
            if len(urls) == before and page > 2:
                # likely reached end of meaningful search pages for this term
                break
            if config.max_listings > 0 and len(urls) >= config.max_listings:
                out = sorted(urls)
                return out[: config.max_listings]
            time.sleep(config.delay)
    out = sorted(urls)
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def discover_urls_with_playwright(config: ScrapeConfig, base_url: str) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    urls: set[str] = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        network_urls: set[str] = set()

        def on_response(resp: Any) -> None:
            try:
                ctype = str(resp.headers.get("content-type") or "").lower()
                if "json" not in ctype and "html" not in ctype and "javascript" not in ctype:
                    return
                text = resp.text()
                found_urls, _ = extract_urls_and_hints_from_payload(text, base_url)
                network_urls.update(found_urls)
            except Exception:
                return

        page.on("response", on_response)
        seen_page_sigs: set[str] = set()
        page_base = base_url if base_url.endswith("/") else (base_url + "/")

        for pageno in range(1, max(2, config.max_pages + 1)):
            page_url = page_base if pageno == 1 else urljoin(page_base, f"page/{pageno}/")
            try:
                page.goto(page_url, wait_until="networkidle", timeout=int(config.timeout * 1000))
            except Exception:
                break

            expand_playwright_results(
                page,
                max_clicks=config.playwright_max_clicks,
                timeout_ms=int(config.timeout * 1000),
            )

            html = page.content()
            discovered = extract_listing_urls(html, page_url)
            sig = hashlib.sha1("||".join(sorted(discovered)).encode("utf-8")).hexdigest() if discovered else ""
            if sig and sig in seen_page_sigs:
                break
            if sig:
                seen_page_sigs.add(sig)

            before = len(urls)
            for u in discovered:
                low = u.lower()
                if "bringatrailer.com/listing/" not in low:
                    continue
                if "land-cruiser" not in low and "lx470" not in low and "lx570" not in low:
                    continue
                urls.add(u.rstrip("/"))
            if len(urls) == before and pageno > 3:
                break
            if config.max_listings > 0 and len(urls) >= config.max_listings:
                break

        context.close()
        browser.close()

    urls.update(network_urls)
    out = sorted(urls)
    if config.max_listings > 0:
        out = out[: config.max_listings]
    return out


def extract_playwright_card_hints(config: ScrapeConfig, base_url: str) -> dict[str, dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    hints: dict[str, dict[str, Any]] = {}

    def parse_card_text(text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        sold = re.search(r"Sold\s+for\s+\$([\d,]+)", text, flags=re.I)
        bid_to = re.search(r"Bid\s+to\s+\$([\d,]+)", text, flags=re.I)
        bids = re.search(r"\b(\d{1,4})\s+bids?\b", text, flags=re.I)
        date_m = re.search(
            r"\bon\s+([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text,
            flags=re.I,
        )
        if sold:
            out["sold_price_usd"] = to_int(sold.group(1))
            out["sale_status"] = "sold"
        elif bid_to:
            out["highest_bid_usd"] = to_int(bid_to.group(1))
            out["sale_status"] = "reserve_not_met"
        if bids:
            out["number_of_bids"] = int(bids.group(1))
        if date_m:
            parsed = parse_datetime(date_m.group(1))
            if parsed:
                out["auction_end_datetime_utc"] = parsed
        return out

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        network_hints: dict[str, dict[str, Any]] = {}

        def on_response(resp: Any) -> None:
            try:
                ctype = str(resp.headers.get("content-type") or "").lower()
                if "json" not in ctype and "html" not in ctype and "javascript" not in ctype:
                    return
                text = resp.text()
                _, parsed_hints = extract_urls_and_hints_from_payload(text, base_url)
                for u, h in parsed_hints.items():
                    prev = network_hints.get(u, {})
                    prev.update(h)
                    network_hints[u] = prev
            except Exception:
                return

        page.on("response", on_response)
        page.goto(base_url, wait_until="networkidle", timeout=int(config.timeout * 1000))

        for _ in range(max(0, getattr(config, "playwright_max_clicks", 40))):
            clicked = False
            for sel in [
                "button:has-text('Load More')",
                "button:has-text('Show More')",
                "button:has-text('More Results')",
                "button:has-text('Load Results')",
                "a:has-text('Load More')",
                "a:has-text('Show More')",
                "a:has-text('More Results')",
                "a:has-text('Load Results')",
                "[data-testid='load-more']",
            ]:
                try:
                    loc = page.locator(sel).first
                    if loc.count() > 0 and loc.is_visible():
                        loc.click(timeout=2000)
                        page.wait_for_timeout(900)
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                break

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        card_nodes = soup.select(".featured-listing-content")
        for node in card_nodes:
            a = node.find("a", href=True)
            if not a:
                continue
            href = a.get("href", "").strip()
            if "/listing/" not in href:
                continue
            full = urljoin(base_url, href).split("?")[0].rstrip("/")
            if not re.search(r"/listing/[^/]+$", full):
                continue
            card_text = node.get_text(" ", strip=True)
            parsed = parse_card_text(card_text)

            for attr in [
                "data-featured_listing_ends",
                "data-featured-listing-ends",
                "data-listing-end",
                "data-end",
                "data-auction-end",
            ]:
                raw_end = str(node.get(attr) or "").strip()
                if raw_end:
                    parsed_end = parse_unix_timestamp(raw_end) or parse_datetime(raw_end)
                    if parsed_end:
                        parsed["auction_end_datetime_utc"] = parsed_end
                        break

            if parsed:
                prev = hints.get(full, {})
                prev.update(parsed)
                hints[full] = prev

        # Secondary fallback: anchor-up traversal for cases where class names differ.
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if "/listing/" not in href:
                continue
            full = urljoin(base_url, href).split("?")[0].rstrip("/")
            if not re.search(r"/listing/[^/]+$", full):
                continue
            card_text = ""
            parent = a.parent
            hops = 0
            while parent is not None and hops < 5:
                txt = parent.get_text(" ", strip=True)
                if txt and ("Sold for" in txt or "Bid to" in txt or "bids" in txt.lower()):
                    card_text = txt
                    break
                parent = getattr(parent, "parent", None)
                hops += 1
            parsed = parse_card_text(card_text)
            # Repo signal: BaT cards may include epoch end time in data-featured_listing_ends.
            if parent is not None:
                for node in [parent] + list(parent.find_all(attrs={"data-featured_listing_ends": True})):
                    raw_end = str(node.get("data-featured_listing_ends") or "").strip()
                    if raw_end:
                        parsed_end = parse_unix_timestamp(raw_end)
                        if parsed_end:
                            parsed["auction_end_datetime_utc"] = parsed_end
                            break
            if parsed:
                prev = hints.get(full, {})
                prev.update(parsed)
                hints[full] = prev

        context.close()
        browser.close()

    for u, h in network_hints.items():
        prev = hints.get(u, {})
        prev.update(h)
        hints[u] = prev
    return hints


def iter_json_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from iter_json_nodes(v)
    elif isinstance(node, list):
        for item in node:
            yield from iter_json_nodes(item)


def load_embedded_json(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in soup.find_all("script", type="application/ld+json"):
        raw = (s.string or s.get_text("", strip=True) or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        out.extend(iter_json_nodes(data))
    return out


def parse_listing(
    url: str,
    html: str,
    response_headers: dict[str, Any] | None = None,
    card_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    article_text = ""
    article = soup.find("article")
    if article:
        article_text = article.get_text(" ", strip=True)
    if not article_text:
        main = soup.find("main")
        if main:
            article_text = main.get_text(" ", strip=True)
    meta_desc = ""
    meta_desc_tag = soup.find("meta", attrs={"property": "og:description"})
    if not meta_desc_tag:
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_desc = str(meta_desc_tag.get("content")).strip()

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = " ".join(h1.get_text(" ", strip=True).split())
    if not title:
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()

    year = None
    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", title)
    if m_year:
        year = int(m_year.group(1))

    vin = ""
    m_vin = re.search(r"\bVIN\s*[:#]?\s*([A-HJ-NPR-Z0-9]{5,20})\b", page_text, flags=re.I)
    if m_vin:
        vin = m_vin.group(1).upper()

    mileage = None
    m_miles = re.search(r"\b([\d,.]+\s*k?)\s*miles?\b", page_text, flags=re.I)
    if m_miles:
        mileage = to_float_miles(m_miles.group(1))

    city_state = ""
    m_loc = re.search(r"in\s+([A-Za-z .'-]+,\s*[A-Z]{2})\b", page_text)
    if m_loc:
        city_state = m_loc.group(1).strip()

    # Outcome signals on BaT usually look like: "Sold for $X" or "Bid to $X".
    sold_price = None
    highest_bid = None
    current_bid = None
    reserve_met = None
    sale_status = "unknown"
    hint = card_hint or {}

    # Prefer title + meta description + article/main content for outcome parsing.
    # This avoids depending on global page text that can include unrelated listings.
    outcome_text = " ".join(x for x in [title, meta_desc, article_text] if x)
    full_text = " ".join(x for x in [outcome_text, page_text] if x)
    sold_match = re.search(r"Sold\s+for\s+(?:USD\s*)?\$[\d,]+", outcome_text, flags=re.I)
    bid_to_match = re.search(r"Bid\s+to\s+(?:USD\s*)?\$[\d,]+", outcome_text, flags=re.I)
    withdrawn_match = re.search(r"Withdrawn", outcome_text, flags=re.I)
    live_match = re.search(
        r"\b(live auction|current bid|auction ends|time left|bid:\s*\$[\d,]+)\b",
        outcome_text,
        flags=re.I,
    )
    current_bid_match = re.search(
        r"(?:Current\s+Bid|Bid)\s*:?\s*(?:USD\s*)?\$[\d,]+",
        outcome_text,
        flags=re.I,
    )

    if sold_match:
        sold_price = parse_money(sold_match.group(0))
        reserve_met = True
        sale_status = "sold"
    elif bid_to_match:
        highest_bid = parse_money(bid_to_match.group(0))
        reserve_met = False
        sale_status = "reserve_not_met"
    elif withdrawn_match:
        sale_status = "withdrawn"
        reserve_met = False
    elif live_match and not sold_match and not bid_to_match:
        sale_status = "live"
        reserve_met = None
        if current_bid_match:
            current_bid = parse_money(current_bid_match.group(0))
    elif current_bid_match and not sold_match and not bid_to_match:
        sale_status = "live"
        current_bid = parse_money(current_bid_match.group(0))

    bids_count = None
    m_bids = re.search(r"\b([0-9]{1,4})\s+bids?\b", page_text, flags=re.I)
    if m_bids:
        bids_count = int(m_bids.group(1))

    end_dt = ""
    json_nodes = load_embedded_json(soup)
    date_candidates: list[tuple[int, str]] = []

    def node_relevance_score(node: dict[str, Any]) -> int:
        score = 0
        node_url = str(node.get("url") or "").strip()
        if node_url and (url.rstrip("/") in node_url.rstrip("/") or node_url.rstrip("/") in url.rstrip("/")):
            score += 4
        name = str(node.get("name") or "").lower()
        if title and name:
            title_tokens = [t for t in re.findall(r"[a-z0-9]+", title.lower()) if len(t) > 2]
            shared = sum(1 for t in set(title_tokens[:8]) if t in name)
            score += min(shared, 3)
        typ = str(node.get("@type") or "").lower()
        if any(k in typ for k in ["product", "article", "event"]):
            score += 1
        return score

    for node in json_nodes:
        score = node_relevance_score(node)
        if isinstance(node.get("endDate"), str):
            parsed = parse_datetime(node.get("endDate"))
            if parsed:
                date_candidates.append((score + 3, parsed))
        if isinstance(node.get("datePublished"), str):
            parsed = parse_datetime(node.get("datePublished"))
            if parsed:
                date_candidates.append((score + 1, parsed))
        if isinstance(node.get("dateModified"), str):
            parsed = parse_datetime(node.get("dateModified"))
            if parsed:
                date_candidates.append((score, parsed))

        # Do not infer sold outcome from generic offers.price: for active listings this
        # is often just current bid.

    if date_candidates:
        # Prefer listing-relevant candidates and then earliest plausible date to avoid
        # grabbing generic site-wide future dates.
        date_candidates.sort(key=lambda x: (-x[0], x[1]))
        top_score = date_candidates[0][0]
        top = [d for s, d in date_candidates if s == top_score]
        top_sorted = sorted(top)
        if top_sorted:
            end_dt = top_sorted[0]

    if not end_dt:
        for meta_key in [
            ("meta", {"property": "article:published_time"}, "content"),
            ("meta", {"name": "date"}, "content"),
            ("meta", {"property": "og:updated_time"}, "content"),
        ]:
            tag = soup.find(meta_key[0], attrs=meta_key[1])
            if tag and tag.get(meta_key[2]):
                end_dt = parse_datetime(tag.get(meta_key[2]))
                if end_dt:
                    break

    if not end_dt:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag and time_tag.get("datetime"):
            end_dt = parse_datetime(time_tag.get("datetime"))

    if not end_dt:
        # Broader fallback: parse any <time> tag text/datetime and pick a plausible past date.
        time_candidates: list[str] = []
        for t in soup.find_all("time"):
            dt_attr = str(t.get("datetime") or "").strip()
            txt = t.get_text(" ", strip=True)
            for raw in [dt_attr, txt]:
                parsed = parse_datetime(raw)
                if parsed:
                    time_candidates.append(parsed)
        if time_candidates:
            # Choose most recent parsed time not in the future.
            now_iso = datetime.now(timezone.utc).isoformat()
            past = [x for x in time_candidates if x <= now_iso]
            if past:
                end_dt = sorted(past)[-1]

    if not end_dt:
        freeform_patterns = [
            r"(?:Sold\s+for|Bid\s+to)\s+\$[\d,]+\s+on\s+([^.;|]+)",
            r"(?:Auction\s+)?Ended\s+([^.;|]+)",
            r"(?:ended|closing)\s+on\s+([^.;|]+)",
        ]
        for pat in freeform_patterns:
            m = re.search(pat, outcome_text, flags=re.I)
            if not m:
                continue
            parsed = parse_datetime(m.group(1))
            if parsed:
                end_dt = parsed
                break

    if not end_dt:
        # Common ended-auction phrase patterns on BaT listing pages.
        patterns = [
            r"(?:Sold\s+for|Bid\s+to)\s+\$[\d,]+\s+on\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
            r"(?:Sold\s+for|Bid\s+to)\s+\$[\d,]+\s+on\s+([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{2,4})",
            r"(?:Sold\s+for|Bid\s+to)\s+\$[\d,]+\s+on\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:Auction\s+)?Ended\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}(?:\s+at\s+\d{1,2}:\d{2}\s*[APMapm]{2})?)",
            r"(?:on\s+Bring\s+a\s+Trailer)[^A-Za-z0-9]{0,20}([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, page_text, flags=re.I)
            if m:
                parsed = parse_datetime(m.group(1))
                if parsed:
                    end_dt = parsed
                    break

    if not end_dt:
        m_ended = re.search(r"Ends\s+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}[^.]{0,50})", page_text)
        if m_ended:
            end_dt = parse_datetime(m_ended.group(1))

    if not end_dt:
        # Try to extract date-like fields from inline script blobs where JSON parsing fails.
        script_blob = " ".join(s.get_text(" ", strip=True) for s in soup.find_all("script"))
        script_patterns = [
            r'"endDate"\s*:\s*"([^"]+)"',
            r'"auction_end(?:_date|Date)?"\s*:\s*"([^"]+)"',
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'"endedAt"\s*:\s*"([^"]+)"',
            r'"closed_at"\s*:\s*"([^"]+)"',
            r'"ends_at"\s*:\s*"([^"]+)"',
            r'"endDate"\s*:\s*(\d{10,13})',
            r'"endedAt"\s*:\s*(\d{10,13})',
            r'"closed_at"\s*:\s*(\d{10,13})',
            r'"ends_at"\s*:\s*(\d{10,13})',
        ]
        for pat in script_patterns:
            m = re.search(pat, script_blob, flags=re.I)
            if not m:
                continue
            raw = m.group(1).strip()
            if raw.isdigit() and len(raw) >= 10:
                try:
                    ts = int(raw[:10])
                    parsed = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                except Exception:
                    parsed = ""
            else:
                parsed = parse_datetime(raw)
            if parsed:
                end_dt = parsed
                break

    if not end_dt and response_headers:
        last_modified = str(response_headers.get("Last-Modified") or "").strip()
        if last_modified:
            end_dt = parse_datetime(last_modified)

    if sale_status == "unknown" and isinstance(hint.get("sale_status"), str):
        sale_status = str(hint.get("sale_status"))
    if sold_price is None and hint.get("sold_price_usd") is not None:
        sold_price = to_int(str(hint.get("sold_price_usd")))
        if sold_price is not None:
            sale_status = "sold"
            reserve_met = True
    if highest_bid is None and hint.get("highest_bid_usd") is not None:
        highest_bid = to_int(str(hint.get("highest_bid_usd")))
        if highest_bid is not None and sale_status == "unknown":
            sale_status = "reserve_not_met"
            reserve_met = False
    if bids_count is None and hint.get("number_of_bids") is not None:
        bids_count = to_int(str(hint.get("number_of_bids")))
    if not end_dt and isinstance(hint.get("auction_end_datetime_utc"), str):
        end_dt = parse_datetime(str(hint.get("auction_end_datetime_utc")))

    if sale_status != "sold":
        sold_price = None

    make = "Toyota" if re.search(r"\bToyota\b", title, flags=re.I) else ""
    model = "Land Cruiser" if re.search(r"\bLand\s+Cruiser\b", title, flags=re.I) else ""

    return {
        "url": url,
        "title": title,
        "make": make,
        "model": model,
        "year": year,
        "vin": vin,
        "mileage": mileage,
        "location": city_state,
        "sale_status": sale_status,
        "reserve_met": reserve_met,
        "sold_price_usd": sold_price,
        "highest_bid_usd": highest_bid,
        "current_bid_usd": current_bid,
        "auction_end_datetime_utc": end_dt,
        "number_of_bids": bids_count,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "url",
        "title",
        "make",
        "model",
        "year",
        "vin",
        "mileage",
        "location",
        "sale_status",
        "reserve_met",
        "sold_price_usd",
        "highest_bid_usd",
        "auction_end_datetime_utc",
        "number_of_bids",
        "scraped_at_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def filter_by_date(
    rows: list[dict[str, Any]],
    start_year: int,
    end_date_utc: datetime,
    strict_end_date: bool = False,
) -> list[dict[str, Any]]:
    start_dt = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    in_range: list[dict[str, Any]] = []
    missing_or_unparseable: list[dict[str, Any]] = []
    out_of_range: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("auction_end_datetime_utc")
        if not value:
            missing_or_unparseable.append(row)
            continue
        try:
            dt = dtparser.parse(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except Exception:
            missing_or_unparseable.append(row)
            continue
        if start_dt <= dt <= end_date_utc:
            in_range.append(row)
        else:
            out_of_range.append(row)

    if strict_end_date:
        return in_range

    if in_range:
        return in_range + missing_or_unparseable

    # If date parsing/range matching yields nothing, keep all rows so downstream
    # model training can still proceed; caller can enable --strict-end-date to enforce.
    return rows


def scrape(config: ScrapeConfig, base_url: str) -> list[dict[str, Any]]:
    if config.discovery_method == "playwright":
        search_rows: list[dict[str, Any]] = []
        series_rows: list[dict[str, Any]] = []
        try:
            search_rows = discover_rows_from_results_search_playwright(config)
        except Exception as exc:
            print(f"WARN: results search discovery failed: {exc}")
        try:
            series_rows = discover_rows_with_playwright(config, base_url)
        except Exception as exc:
            print(f"WARN: series-page playwright discovery failed: {exc}")

        merged: dict[str, dict[str, Any]] = {}
        # Results-search rows are primary; series rows fill gaps.
        for row in search_rows + series_rows:
            merged[row["url"]] = row
        combined_rows = list(merged.values())
        print("Discovery method used: playwright")
        print(f"Playwright results-search rows extracted: {len(search_rows)}")
        print(f"Playwright series rows extracted: {len(series_rows)}")
        print(f"Playwright combined unique rows: {len(combined_rows)}")
        return combined_rows

    listing_urls: list[str] = []
    method_counts: dict[str, int] = {}
    card_hints: dict[str, dict[str, Any]] = {}

    if config.discovery_method == "auto":
        aggregate: set[str] = set()
        for method_name, fn in [
            ("pages", lambda: discover_urls_from_pages(config, base_url)),
            ("playwright", lambda: discover_urls_with_playwright(config, base_url)),
            ("search", lambda: discover_urls_from_search(config, base_url)),
            ("wpjson", lambda: discover_urls_from_wp_json(config, base_url)),
            ("sitemap", lambda: discover_urls_from_sitemap(config)),
        ]:
            try:
                discovered = fn()
            except Exception as exc:
                print(f"WARN: discovery method {method_name} failed: {exc}")
                discovered = []
            method_counts[method_name] = len(discovered)
            aggregate.update(discovered)
            if config.max_listings > 0 and len(aggregate) >= config.max_listings:
                break
        listing_urls = sorted(aggregate)
        if config.max_listings > 0:
            listing_urls = listing_urls[: config.max_listings]
    elif config.discovery_method == "pages":
        listing_urls = discover_urls_from_pages(config, base_url)
        method_counts["pages"] = len(listing_urls)
    elif config.discovery_method == "search":
        listing_urls = discover_urls_from_search(config, base_url)
        method_counts["search"] = len(listing_urls)
    elif config.discovery_method == "playwright":
        listing_urls = discover_urls_with_playwright(config, base_url)
        method_counts["playwright"] = len(listing_urls)
    elif config.discovery_method == "wpjson":
        listing_urls = discover_urls_from_wp_json(config, base_url)
        method_counts["wpjson"] = len(listing_urls)
    elif config.discovery_method == "sitemap":
        listing_urls = discover_urls_from_sitemap(config)
        method_counts["sitemap"] = len(listing_urls)

    print(f"Discovery method used: {config.discovery_method}")
    for k, v in method_counts.items():
        print(f"  - {k}: {v}")
    print(f"Listing URLs discovered: {len(listing_urls)}")
    if config.discovery_method in {"auto", "playwright"} and listing_urls:
        try:
            card_hints = extract_playwright_card_hints(config, base_url)
            print(f"Playwright card hints extracted: {len(card_hints)}")
            hints_with_dates = sum(
                1 for h in card_hints.values() if str(h.get("auction_end_datetime_utc") or "").strip()
            )
            print(f"Playwright hints with end datetime: {hints_with_dates}")
            # Guardrail: if all hint dates collapse to one future timestamp, ignore them.
            hint_dates = [str(h.get("auction_end_datetime_utc") or "").strip() for h in card_hints.values()]
            hint_dates = [d for d in hint_dates if d]
            if hint_dates:
                uniq_dates = sorted(set(hint_dates))
                if len(uniq_dates) == 1:
                    try:
                        one_dt = dtparser.parse(uniq_dates[0]).astimezone(timezone.utc)
                        if one_dt > config.end_date_utc:
                            for u in list(card_hints.keys()):
                                card_hints[u].pop("auction_end_datetime_utc", None)
                            print("WARN: Ignored single shared future hint date across listings.")
                    except Exception:
                        pass
        except Exception as exc:
            print(f"WARN: failed to extract playwright card hints: {exc}")

    detail_fetch_mode = config.detail_fetch
    if detail_fetch_mode == "auto":
        detail_fetch_mode = "playwright" if config.discovery_method == "playwright" else "requests"
    print(f"Detail fetch mode: {detail_fetch_mode}")

    rows: list[dict[str, Any]] = []
    if detail_fetch_mode == "playwright":
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            print(f"WARN: playwright detail fetch unavailable, falling back to requests: {exc}")
            detail_fetch_mode = "requests"

    if detail_fetch_mode == "playwright":
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            for idx, listing_url in enumerate(listing_urls, start=1):
                try:
                    page.goto(listing_url, wait_until="networkidle", timeout=int(config.timeout * 1000))
                    listing_html = page.content()
                    row = parse_listing(
                        listing_url,
                        listing_html,
                        response_headers={},
                        card_hint=card_hints.get(listing_url.rstrip("/"), {}),
                    )
                    rows.append(row)
                except Exception as exc:
                    print(f"WARN: failed to parse {listing_url}: {exc}")
                if idx < len(listing_urls):
                    time.sleep(config.delay)
            context.close()
            browser.close()
    else:
        for idx, listing_url in enumerate(listing_urls, start=1):
            try:
                listing_resp = fetch_response(listing_url, timeout=config.timeout)
                row = parse_listing(
                    listing_url,
                    listing_resp.text,
                    response_headers=dict(listing_resp.headers),
                    card_hint=card_hints.get(listing_url.rstrip("/"), {}),
                )
                rows.append(row)
            except Exception as exc:
                print(f"WARN: failed to parse {listing_url}: {exc}")

            if idx < len(listing_urls):
                time.sleep(config.delay)

    return rows


def main() -> int:
    args = parse_args()
    config = ScrapeConfig(
        start_year=args.start_year,
        end_date_utc=datetime.now(timezone.utc),
        timeout=args.timeout,
        delay=args.delay,
        max_pages=args.max_pages,
        max_listings=args.max_listings,
        discovery_method=args.discovery_method,
        sitemap_index_url=args.sitemap_index_url,
        wpjson_max_pages=args.wpjson_max_pages,
        playwright_max_clicks=args.playwright_max_clicks,
        detail_fetch=args.detail_fetch,
        results_search_terms=args.results_search_terms,
    )

    seed_urls = args.base_urls if args.base_urls else [args.base_url]
    all_rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for seed_url in seed_urls:
        try:
            rows_for_seed = scrape(config=config, base_url=seed_url)
        except Exception as exc:
            print(f"WARN: failed to scrape seed URL {seed_url}: {exc}")
            continue
        for row in rows_for_seed:
            row_url = str(row.get("url") or "").strip()
            if not row_url:
                continue
            if row_url in seen_urls:
                continue
            seen_urls.add(row_url)
            all_rows.append(row)

    raw_rows = all_rows
    raw_dates = [str(r.get("auction_end_datetime_utc") or "").strip() for r in raw_rows]
    raw_dates = [d for d in raw_dates if d]
    print(f"Raw rows with end datetime: {len(raw_dates)}")
    if raw_dates:
        print(f"Raw unique end datetimes: {len(set(raw_dates))}")
        print(f"Raw min end datetime: {min(raw_dates)}")
        print(f"Raw max end datetime: {max(raw_dates)}")
    rows = filter_by_date(
        raw_rows,
        start_year=config.start_year,
        end_date_utc=config.end_date_utc,
        strict_end_date=args.strict_end_date,
    )

    out_json = Path(args.json)
    out_csv = Path(args.csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_csv(out_csv, rows)

    print(f"Raw rows parsed: {len(raw_rows)}")
    print(f"Rows saved after date filter: {len(rows)}")
    non_empty_dates = [str(r.get("auction_end_datetime_utc") or "").strip() for r in rows]
    non_empty_dates = [d for d in non_empty_dates if d]
    print(f"Rows with end datetime: {len(non_empty_dates)}")
    if non_empty_dates:
        print(f"Unique end datetimes: {len(set(non_empty_dates))}")
        print(f"Min end datetime: {min(non_empty_dates)}")
        print(f"Max end datetime: {max(non_empty_dates)}")
    if raw_rows and not rows:
        print("WARN: Date filter removed all rows. Re-run without --strict-end-date or inspect date parsing.")
    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
