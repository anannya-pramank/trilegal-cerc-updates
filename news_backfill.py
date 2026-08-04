#!/usr/bin/env python3
"""
news_backfill.py — one-year (or arbitrary window) historical backfill for the
WordPress-based sources (Saur, Energetica, Mercom).

Why a separate script from news_watch.py
-----------------------------------------
RSS only carries a site's most recent items (~days to weeks). A year of history
needs the site's ARCHIVE. All three renewable-trade sites run WordPress, which
exposes a REST API that paginates the full post history with dates:

    /wp-json/wp/v2/posts?after=<ISO>&before=<ISO>&per_page=100&page=N&_fields=...

So this script reuses everything downstream identically — same admission gate,
same news_items schema, same MiniLM embedding at ingest, same dedup-by-URL — and
only swaps the acquisition method (dated WP pagination instead of RSS). Tag/link/
export treat a backfilled row exactly like a live one.

ET EnergyWorld and Moneycontrol are intentionally NOT here: no public archive API,
and (Moneycontrol) mostly markets noise. They contribute recent history via RSS
(news_watch.py) only.

Resumability
------------
Walks month by month, newest first. Commits after each month. --deadline-minutes
gives a soft wall-clock cap so an Actions run flushes and exits 0 before the 6h
kill; re-run with the same window to resume (dedup makes already-ingested months
cheap no-ops). Per-site --state lets you see how far each got.

Usage
-----
  # default: trailing 12 months for the three WP sites
  DATABASE_URL=... python news_backfill.py

  python news_backfill.py --months 12
  python news_backfill.py --since 2024-08-01 --until 2025-08-01
  python news_backfill.py --source saur            # one site
  python news_backfill.py --no-embed               # ingest now, embed/tag later
  python news_backfill.py --dry-run                # validate WP endpoints, no write
  python news_backfill.py --deadline-minutes 300   # for Actions

Deps: requests, sentence-transformers, pyyaml, psycopg2-binary, pgvector.
Reuses news_watch.py for the gate, cleaning, embedding, and upsert.
"""
import argparse
import datetime as dt
import sys
import time

import requests

import news_watch as NW   # reuse gate, clean(), item_id(), embed_rows(), upsert(), connect()


def log(m):
    print(m, file=sys.stderr, flush=True)


# WordPress REST base per source id. If a site's API path differs, change here.
# (Standard WP is /wp-json/wp/v2/posts; some installs sit under a subpath.)
WP_BASE = {
    "saur":       "https://www.saurenergy.com/wp-json/wp/v2/posts",
    "energetica": "https://www.energetica-india.net/wp-json/wp/v2/posts",
    "mercom":     "https://www.mercomindia.com/wp-json/wp/v2/posts",
}

WP_SOURCES = set(WP_BASE)

HEADERS = {"User-Agent": "cerc-atlas-backfill/1.0 (+regulatory research)"}


def month_windows(since, until):
    """Yield (after_iso, before_iso, label) month spans, newest first."""
    spans = []
    cur = dt.date(until.year, until.month, 1)
    while cur >= dt.date(since.year, since.month, 1):
        # end of this month = start of next
        if cur.month == 12:
            nxt = dt.date(cur.year + 1, 1, 1)
        else:
            nxt = dt.date(cur.year, cur.month + 1, 1)
        after = max(cur, since)
        before = min(nxt, until + dt.timedelta(days=1))
        spans.append((after.isoformat() + "T00:00:00",
                      before.isoformat() + "T00:00:00",
                      cur.strftime("%Y-%m")))
        # step back one month
        if cur.month == 1:
            cur = dt.date(cur.year - 1, 12, 1)
        else:
            cur = dt.date(cur.year, cur.month - 1, 1)
    return spans


def fetch_wp_month(base, after_iso, before_iso, dry_run=False):
    """All posts in [after,before). Returns list of raw dicts, or None on hard fail."""
    out = []
    page = 1
    while True:
        params = {
            "after": after_iso, "before": before_iso,
            "per_page": 100, "page": page, "orderby": "date", "order": "desc",
            "_fields": "id,link,title,excerpt,date",
        }
        try:
            r = requests.get(base, params=params, headers=HEADERS, timeout=30)
        except Exception as e:
            log(f"    request error p{page}: {e}")
            return None if page == 1 else out
        if r.status_code == 400 and page > 1:
            break                         # WP returns 400 past the last page
        if r.status_code == 404:
            log(f"    404 — WP REST not at this base?")
            return None
        if r.status_code != 200:
            log(f"    HTTP {r.status_code} p{page}")
            break
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        total_pages = r.headers.get("X-WP-TotalPages")
        if total_pages and page >= int(total_pages):
            break
        page += 1
        if page > 30:                     # safety: 3000 posts/month is implausible
            break
        if not dry_run:
            time.sleep(0.3)               # be polite to the origin
    return out


def raw_to_row(post, source):
    url = post.get("link")
    if not url:
        return None
    title = NW.clean((post.get("title") or {}).get("rendered"))
    summary = NW.clean((post.get("excerpt") or {}).get("rendered"))
    if not title:
        return None
    date = (post.get("date") or "")[:10] or None
    return {"id": NW.item_id(url), "source": source, "url": url,
            "title": title, "summary": summary, "published": date}


def backfill_source(sid, since, until, dry_run, embed, deadline_ts):
    src = next(s for s in NW.load_sources("news_sources.yaml") if s["id"] == sid)
    base = WP_BASE[sid]
    log(f"[{sid}] WP backfill {since} … {until}  ({base})")
    total_admitted = 0
    for after_iso, before_iso, label in month_windows(since, until):
        if deadline_ts and time.monotonic() > deadline_ts:
            log(f"[{sid}] deadline hit at {label}; committed so far. Re-run to resume.")
            return total_admitted, True
        posts = fetch_wp_month(base, after_iso, before_iso, dry_run)
        if posts is None:
            log(f"[{sid}] {label}: WP endpoint unavailable — skipping source")
            return total_admitted, False
        rows = []
        for p in posts:
            row = raw_to_row(p, sid)
            if not row:
                continue
            admitted, score, hits = NW.relevance_and_gate(
                row["title"], row.get("summary"), src)
            if not admitted:
                continue
            row["relevance"] = score
            rows.append(row)
        log(f"[{sid}] {label}: {len(posts)} posts -> {len(rows)} admitted")
        total_admitted += len(rows)
        if dry_run or not rows:
            continue
        if embed:
            NW.embed_rows(rows)
        conn = NW.connect()
        NW.upsert(conn, rows, embed)
        conn.close()
    return total_admitted, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(WP_SOURCES),
                    help="one WP source; default = all three")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--since")   # ISO date, overrides --months
    ap.add_argument("--until")   # ISO date, default today
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline-minutes", type=int, default=0)
    args = ap.parse_args()

    until = dt.date.fromisoformat(args.until) if args.until else dt.date.today()
    if args.since:
        since = dt.date.fromisoformat(args.since)
    else:
        # go back N months
        y, m = until.year, until.month - args.months
        while m <= 0:
            m += 12; y -= 1
        since = dt.date(y, m, 1)

    sources = [args.source] if args.source else sorted(WP_SOURCES)
    embed = not args.no_embed
    deadline_ts = (time.monotonic() + args.deadline_minutes * 60
                   ) if args.deadline_minutes else 0

    grand = 0
    for sid in sources:
        n, stopped = backfill_source(sid, since, until, args.dry_run, embed, deadline_ts)
        grand += n
        if stopped:
            break
    log(f"backfill {'(dry-run) ' if args.dry_run else ''}total admitted: {grand}")
    if args.dry_run:
        log("dry-run: nothing written. If counts look right, drop --dry-run.")


if __name__ == "__main__":
    main()
