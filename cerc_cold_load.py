#!/usr/bin/env python3
"""
cerc_cold_load.py — one-shot backfill of a full year of CERC orders into the
existing Supabase `cerc_orders` table. Sibling of aptel_cold_load.py.

Why not just run the daily scraper with CERC_ORDERS_YEAR? Because the daily
pipeline has side effects a backfill must NOT have:
  * it appends thousands of rows to cerc/cerc_orders_master.csv (the git ledger),
  * it overwrites cerc/cerc_orders_new.json — the Power Automate feed — which
    would blast a decade of stale orders into the Teams channel,
  * its dedup is the CSV ledger, not the database.

This script instead reuses the scraper's proven plumbing (www-host fetch with
curl TLS fallback, pymupdf4llm extraction with use_ocr=False, the semantic
digest ranker) but dedups directly against `cerc_orders` and upserts with the
exact same SQL contract as load_cerc.py. It touches no files in cerc/ that the
daily pipeline reads.

Resumability: id = sha1(pdf_url)[:16] (same scheme everywhere), existing ids
are skipped up front, and rows commit in batches — a crash or the GitHub 6h
kill loses at most one batch. Re-run the same year freely.

Usage (PowerShell):
    $env:DATABASE_URL = "postgresql://postgres.<ref>:<pw>@<host>:6543/postgres"
    python cerc_cold_load.py --year 2019 --dry-run     # list + report, write nothing
    python cerc_cold_load.py --year 2019               # full backfill of that year
    python cerc_cold_load.py --year 2019 --limit 5     # smoke test
    python cerc_cold_load.py --year 2019 --deadline-minutes 320   # for Actions
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the daily scraper's plumbing: fetch (www host + curl TLS fallback with
# retry/backoff), _pdf_to_text (pymupdf4llm use_ocr=False -> pdfplumber),
# make_digest (semantic MiniLM ranking), make_id, and the orders-table parser.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrapers import cerc_scraper as cs  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim; matches cerc_orders vector(384)
EMBED_DIM = 384

_st_model = None


def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (first run downloads ~80MB) …")
        _st_model = SentenceTransformer(MODEL_NAME)
        # Share the instance with the scraper's digest ranker so we don't load
        # MiniLM twice (make_digest -> _semantic_rank caches on this attribute).
        cs._semantic_rank._model = _st_model
    return _st_model


def embed_to_literal(text: str) -> str:
    """Embed one string -> pgvector literal. Mirrors load_cerc.py: the
    embedding is on the DIGEST, normalized."""
    vec = _get_model().encode([text or "(empty)"], normalize_embeddings=True)[0]
    if len(vec) != EMBED_DIM:
        sys.exit(f"Embedding dim {len(vec)} != {EMBED_DIM}; wrong model?")
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def parse_cerc_date(value):
    """DD.MM.YYYY -> date, tolerant of a few variants; None on failure
    (store NULL rather than crash the year on one bad row)."""
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def scrape_orders_for_year(year: int) -> list:
    """List all orders on the year's archive page (recent_orders<year>.html),
    or the live page for the current year. Fails loudly if the page/table is
    missing — no silent fallback to another year (scraper's philosophy)."""
    current = datetime.now().year
    if year == current:
        os.environ.pop("CERC_ORDERS_YEAR", None)
    else:
        os.environ["CERC_ORDERS_YEAR"] = str(year)
    rows = cs.scrape_orders()
    if not rows:
        sys.exit(f"No orders parsed for {year} — archive page missing or layout "
                 f"changed. Check {cs.resolve_orders_url()} in a browser.")
    for r in rows:
        r["id"] = cs.make_id(r["pdf_url"])
    return rows


# ================= DB =================

def db_connect():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL to your Supabase transaction-pooler string "
                 "(remember: '@' in the password must be URL-encoded as %40).")
    return psycopg2.connect(url)


def existing_ids(conn) -> set:
    with conn.cursor() as cur:
        cur.execute("select id from cerc_orders")
        return {r[0] for r in cur.fetchall()}


# Exact same contract as load_cerc.py, including the coalesce that never
# overwrites stored fulltext with NULL.
UPSERT = """
    insert into cerc_orders
        (id, petition_no, subject, date_order, date_posted,
         category, pdf_url, pdf_digest, pdf_fulltext, embedding, scraped_at)
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)
    on conflict (id) do update set
        petition_no  = excluded.petition_no,
        subject      = excluded.subject,
        date_order   = excluded.date_order,
        date_posted  = excluded.date_posted,
        category     = excluded.category,
        pdf_url      = excluded.pdf_url,
        pdf_digest   = excluded.pdf_digest,
        pdf_fulltext = coalesce(excluded.pdf_fulltext, cerc_orders.pdf_fulltext),
        embedding    = excluded.embedding,
        scraped_at   = excluded.scraped_at;
"""


# ================= MAIN =================

def main():
    ap = argparse.ArgumentParser(description="Cold-load a year of CERC orders into Supabase.")
    ap.add_argument("--year", type=int, default=datetime.now().year)
    ap.add_argument("--dry-run", action="store_true",
                    help="Scrape the listing and report; download nothing, write nothing.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N new orders (0 = all).")
    ap.add_argument("--max-pdf-pages", type=int, default=120,
                    help="Per-PDF page cap (matches the daily scraper's MAX_PDF_PAGES).")
    ap.add_argument("--request-delay", type=float, default=1.0,
                    help="Seconds between PDF fetches — CERC rate-limits rapid requests.")
    ap.add_argument("--batch", type=int, default=20, help="Rows per commit.")
    ap.add_argument("--deadline-minutes", type=int, default=0,
                    help="Soft wall-clock cap: stop cleanly (flush + exit 0) after N "
                         "minutes so GitHub's 6h job kill never hits mid-batch. "
                         "0 = no deadline. Re-running resumes where this left off.")
    ap.add_argument("--fulltext-dir", default="",
                    help="Optionally also write full text to <dir>/<id>.md for local "
                         "audit. Default off — the DB column is the source of truth, "
                         "and the backfill must not touch the committed cerc/ tree.")
    args = ap.parse_args()

    # Respect the daily scraper's page cap knob without editing its module.
    cs.MAX_PDF_PAGES = args.max_pdf_pages

    print(f"Scraping CERC orders archive for {args.year} …")
    listing = scrape_orders_for_year(args.year)
    print(f"  {len(listing)} orders listed for {args.year}")

    already = set()
    if not args.dry_run:
        conn = db_connect()
        already = existing_ids(conn)
        conn.close()

    todo = [r for r in listing if r["id"] not in already]
    print(f"  {len(already)} ids already in cerc_orders; {len(todo)} candidates remain")
    if args.limit:
        todo = todo[: args.limit]

    if args.dry_run:
        dates = sorted(d for d in (parse_cerc_date(r["date_order"]) for r in listing) if d)
        print("\n--- DRY RUN ---")
        print(f"  year:            {args.year}")
        print(f"  listed:          {len(listing)}")
        print(f"  would download:  {len(todo)}")
        if dates:
            print(f"  order dates:     {dates[0]} … {dates[-1]}")
        print("\n  first few:")
        for r in todo[:10]:
            print(f"    {r['date_order']:>10}  {r['petition_no'][:70]}")
        print("\nNo PDFs fetched, no writes. Re-run without --dry-run to load.")
        return

    if not todo:
        print("Nothing to load. Done.")
        return

    if args.fulltext_dir:
        os.makedirs(args.fulltext_dir, exist_ok=True)

    conn = db_connect()
    batch, loaded, skipped = [], 0, 0

    def flush():
        nonlocal batch
        if not batch:
            return
        with conn.cursor() as cur:
            cur.executemany(UPSERT, batch)
        conn.commit()
        batch = []

    try:
        started = time.monotonic()
        for i, r in enumerate(todo, 1):
            if args.deadline_minutes and (time.monotonic() - started) > args.deadline_minutes * 60:
                print(f"\n[deadline] {args.deadline_minutes} min reached after {i-1}/{len(todo)} "
                      f"orders — flushing and exiting cleanly. Re-run to resume.")
                break
            print(f"[{i}/{len(todo)}] {r['petition_no'][:70]}")
            try:
                content = cs.fetch(r["pdf_url"], binary=True)
            except Exception as e:
                print(f"    download failed, skipping (will retry next run): {e}")
                skipped += 1
                continue

            try:
                full_text = cs._pdf_to_text(content)
            except Exception as e:
                print(f"    extraction failed, skipping (will retry next run): {e}")
                skipped += 1
                continue
            if not full_text:
                print("    no extractable text, skipping")
                skipped += 1
                continue

            if args.fulltext_dir:
                with open(os.path.join(args.fulltext_dir, f"{r['id']}.md"),
                          "w", encoding="utf-8") as f:
                    f.write(full_text)

            digest = cs.make_digest(full_text)
            embedding = embed_to_literal(digest)

            batch.append((
                r["id"], r["petition_no"], r["subject"],
                parse_cerc_date(r["date_order"]), parse_cerc_date(r["date_posted"]),
                r["category"], r["pdf_url"], digest, full_text, embedding,
                datetime.now(timezone.utc).isoformat(),
            ))
            loaded += 1
            if len(batch) >= args.batch:
                flush()
                print(f"    … committed (running total {loaded})")
            if args.request_delay:
                time.sleep(args.request_delay)
    finally:
        flush()
        conn.close()

    print(f"\nDone. Loaded {loaded}, skipped {skipped} (deferred to a re-run).")


if __name__ == "__main__":
    main()
