#!/usr/bin/env python3
"""
news_backfill_sitemap.py — history backfill via XML sitemaps.

The probe showed WP REST is not usable for these three sites (Saur returns HTML,
Energetica has no REST, Mercom's REST is 403). But ALL THREE expose XML sitemaps
(200). Sitemaps are the robust path: they're built to be crawled, list the whole
archive with last-modified dates, and dodge both the REST block and feed-
pagination quirks.

Method
------
1. Fetch the site's sitemap. If it's a sitemap INDEX (points to child sitemaps),
   recurse into children, preferring ones that look post/news/article-ish and
   whose date hints fall in-window.
2. Collect article URLs with their <lastmod> (or news:publication_date) dates,
   filtered to [since, until].
3. For each URL, fetch the page and read metadata from OG/article meta tags
   (og:title, og:description, article:published_time) — no full-body scrape.
4. Push each item through the SAME pipeline as everything else: news_watch's
   admission gate, MiniLM embedding, dedup-by-URL, upsert into news_items.

Politeness + resumability
-------------------------
Rate-limited (--delay). Resumable via --deadline-minutes (flush + exit 0 before
the Actions kill; re-run to continue — dedup makes seen URLs cheap). Per-source.

Usage
-----
  DATABASE_URL=... python news_backfill_sitemap.py               # all 3, 12 months
  python news_backfill_sitemap.py --source energetica
  python news_backfill_sitemap.py --months 12 --delay 0.5
  python news_backfill_sitemap.py --dry-run                       # list, don't write
  python news_backfill_sitemap.py --deadline-minutes 300          # Actions

Deps: requests, sentence-transformers, pyyaml, psycopg2-binary, pgvector.
Reuses news_watch for gate/clean/embed/upsert/connect.
"""
import argparse
import datetime as dt
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

import news_watch as NW


def log(m):
    print(m, file=sys.stderr, flush=True)


# Entry sitemaps per source (from the probe: these returned 200 XML).
SITEMAPS = {
    "saur":       ["https://www.saurenergy.com/sitemap.xml",
                   "https://www.saurenergy.com/news-sitemap.xml"],
    "energetica": ["https://www.energetica-india.net/sitemap.xml"],
    "mercom":     ["https://www.mercomindia.com/sitemap_index.xml",
                   "https://www.mercomindia.com/news-sitemap.xml"],
}

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
HEADERS = {"User-Agent": BROWSER_UA, "Accept": "*/*"}

# child-sitemap URL hints worth recursing into (skip tag/category/author maps)
GOOD_CHILD = re.compile(r"(post|news|article|sitemap-pt)", re.I)
SKIP_CHILD = re.compile(r"(tag|category|author|page-sitemap|product|"
                        r"attachment|media|image|video|user)", re.I)

SM_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


def get(url, timeout=25):
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as e:
        log(f"    GET error {url}: {e}")
        return None


def parse_date(s):
    if not s:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_sitemap(xml_bytes):
    """Return ('index'|'urlset', [ (loc, lastmod_date) ])."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log(f"    sitemap parse error: {e}")
        return None, []
    tag = root.tag.lower()
    entries = []
    if tag.endswith("sitemapindex"):
        for sm in root.findall("sm:sitemap", SM_NS):
            loc = sm.findtext("sm:loc", default="", namespaces=SM_NS)
            lm = parse_date(sm.findtext("sm:lastmod", default="", namespaces=SM_NS))
            if loc:
                entries.append((loc.strip(), lm))
        return "index", entries
    # urlset
    for u in root.findall("sm:url", SM_NS):
        loc = u.findtext("sm:loc", default="", namespaces=SM_NS)
        lm = parse_date(u.findtext("sm:lastmod", default="", namespaces=SM_NS))
        if lm is None:  # news sitemaps carry the date under news:publication_date
            npd = u.find("news:news/news:publication_date", SM_NS)
            if npd is not None:
                lm = parse_date(npd.text)
        if loc:
            entries.append((loc.strip(), lm))
    return "urlset", entries


def collect_urls(seeds, since, until, deadline_ts):
    """Walk sitemap(s) -> list of (url, date) within window. Recurses indexes."""
    seen_sm = set()
    urls = {}
    queue = list(seeds)
    while queue:
        if deadline_ts and time.monotonic() > deadline_ts:
            log("    deadline during sitemap crawl; using what we have")
            break
        sm_url = queue.pop(0)
        if sm_url in seen_sm:
            continue
        seen_sm.add(sm_url)
        r = get(sm_url)
        if not r or r.status_code != 200:
            log(f"    sitemap {sm_url}: HTTP {getattr(r,'status_code','ERR')}")
            continue
        kind, entries = parse_sitemap(r.content)
        if kind == "index":
            for loc, lm in entries:
                if SKIP_CHILD.search(loc):
                    continue
                # if child has a lastmod older than window, skip it wholesale
                if lm and lm < since:
                    continue
                if GOOD_CHILD.search(loc) or lm is None:
                    queue.append(loc)
            log(f"    index {sm_url}: {len(entries)} child sitemaps queued")
        else:
            n_in = 0
            for loc, lm in entries:
                if lm is None or (since <= lm <= until):
                    if loc not in urls or (lm and (urls[loc] is None or lm > urls[loc])):
                        urls[loc] = lm
                    n_in += 1
            log(f"    urlset {sm_url}: {len(entries)} urls, {n_in} in-window")
    return urls


META_PATTERNS = {
    "title": [r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
              r'<title[^>]*>([^<]+)</title>'],
    "desc":  [r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)',
              r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)'],
    "date":  [r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)',
              r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']+)'],
}


def extract_meta(html):
    out = {}
    for key, pats in META_PATTERNS.items():
        for p in pats:
            m = re.search(p, html, re.I)
            if m:
                out[key] = m.group(1).strip()
                break
    return out


def backfill(sid, since, until, dry_run, embed, delay, deadline_ts):
    src = next(s for s in NW.load_sources("news_sources.yaml") if s["id"] == sid)
    seeds = SITEMAPS[sid]
    log(f"[{sid}] sitemap backfill {since} … {until}")
    urls = collect_urls(seeds, since, until, deadline_ts)
    log(f"[{sid}] {len(urls)} candidate article URLs in window")

    admitted = 0
    processed = 0
    batch = []
    seen_ids = set()          # url-normalized ids already handled this run
    for url, lm in sorted(urls.items(), key=lambda kv: (kv[1] or dt.date.min), reverse=True):
        if deadline_ts and time.monotonic() > deadline_ts:
            log(f"[{sid}] deadline hit after {processed} pages; flushing.")
            break
        # two URLs can normalize to the same id (utm/slug variants). Skip dups
        # BEFORE fetching — saves a request and prevents the ON CONFLICT dup.
        iid = NW.item_id(url)
        if iid in seen_ids:
            continue
        seen_ids.add(iid)
        processed += 1
        r = get(url)
        if delay and not dry_run:
            time.sleep(delay)
        if not r or r.status_code != 200 or "html" not in r.headers.get("Content-Type", "").lower():
            continue
        meta = extract_meta(r.text)
        title = NW.clean(meta.get("title"))
        summary = NW.clean(meta.get("desc"))
        date = parse_date(meta.get("date")) or lm
        if not title:
            continue
        admitted_flag, score, hits = NW.relevance_and_gate(title, summary, src)
        if not admitted_flag:
            continue
        row = {"id": iid, "source": sid, "url": url, "title": title,
               "summary": summary,
               "published": date.isoformat() if date else None,
               "relevance": score}
        batch.append(row)
        admitted += 1
        # flush in chunks so a deadline mid-run still saves progress
        if len(batch) >= 50 and not dry_run:
            if embed:
                NW.embed_rows(batch)
            conn = NW.connect(); NW.upsert(conn, batch, embed); conn.close()
            log(f"[{sid}] flushed {len(batch)} (total admitted {admitted})")
            batch = []
    if batch and not dry_run:
        if embed:
            NW.embed_rows(batch)
        conn = NW.connect(); NW.upsert(conn, batch, embed); conn.close()
        log(f"[{sid}] flushed final {len(batch)}")
    log(f"[{sid}] done: {processed} pages fetched, {admitted} admitted")
    return admitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SITEMAPS))
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--since"); ap.add_argument("--until")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between page fetches")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline-minutes", type=int, default=0)
    args = ap.parse_args()

    until = dt.date.fromisoformat(args.until) if args.until else dt.date.today()
    if args.since:
        since = dt.date.fromisoformat(args.since)
    else:
        y, m = until.year, until.month - args.months
        while m <= 0:
            m += 12; y -= 1
        since = dt.date(y, m, 1)

    sources = [args.source] if args.source else sorted(SITEMAPS)
    embed = not args.no_embed
    deadline_ts = (time.monotonic() + args.deadline_minutes * 60) if args.deadline_minutes else 0

    grand = 0
    for sid in sources:
        grand += backfill(sid, since, until, args.dry_run, embed, args.delay, deadline_ts)
        if deadline_ts and time.monotonic() > deadline_ts:
            log("global deadline reached; stopping. Re-run to resume.")
            break
    log(f"backfill {'(dry-run) ' if args.dry_run else ''}total admitted: {grand}")


if __name__ == "__main__":
    main()
