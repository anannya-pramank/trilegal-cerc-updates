#!/usr/bin/env python3
"""
aptel_cold_load.py — one-shot backfill of a full year of APTEL judgements/orders
into a self-contained Supabase `aptel_orders` table.

Why a separate table (for now): this commits to NOTHING about the eventual
unified-vs-sibling decision. The table carries the SAME contract as cerc_orders
(petition / title / digest / fulltext / embedding(384) / fts), so later you can
either point a UNION view at it (sibling approach) or `insert ... select` it into
a unified `documents` table — both are a one-liner away. Get the data in first.

What it does, per order:
  1. Pages through old-judgement-data?field_judge_year_value=<year>&page=N
     until no new rows appear (Drupal Views pager).
  2. Downloads each PDF, extracts the COMPLETE text (pymupdf4llm preferred,
     pdfplumber fallback) up to --max-pdf-pages, and stores it verbatim.
  3. Builds a relevance-ranked digest (semantic, MiniLM) for the embedding +
     a skimmable snippet. The embedding is on the digest; FTS covers the full
     text, so keyword search still reaches operative paragraphs in long appeals.
  4. Upserts into aptel_orders, committing per batch so a crash mid-backfill
     loses at most one batch — re-running resumes (existing ids are skipped).

Resumability: the id is sha1(pdf_url)[:16] (same scheme as your aptel_watcher /
cerc_scraper), and the upsert is `on conflict (id) do update`. Re-run freely.

Usage (PowerShell):
    $env:DATABASE_URL = "postgresql://postgres.<ref>:<pw>@<host>:6543/postgres"
    python aptel_cold_load.py --init-db            # create table + indexes once
    python aptel_cold_load.py --dry-run            # scrape listing, report, touch nothing
    python aptel_cold_load.py                      # full backfill of the current year
    python aptel_cold_load.py --year 2025          # a different year
    python aptel_cold_load.py --limit 5            # smoke test: only 5 PDFs
"""

import argparse
import hashlib
import io
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= CONFIG =================

BASE_URL    = "https://aptel.gov.in"
ORDERS_PATH = "/en/old-judgement-data"

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim; matches cerc_orders so vectors are comparable
EMBED_DIM  = 384

HARD_CAP     = 80_000             # ceiling for semantic_extract's selection input
DIGEST_CHARS = 6_000              # size of the embedded/skimmable digest (matches CERC)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

SUMMARY_QUERIES = [
    "What is the final order, decision, or direction issued by the tribunal?",
    "What are the main legal issues and questions decided in this case?",
    "What is the reasoning and legal basis for the decision?",
    "Who are the parties and what is the nature of the dispute or petition?",
]

# Same shape/contract as cerc_orders. hnsw needs no training and gives good recall;
# if your pgvector build prefers ivfflat, swap the embedding index accordingly.
DDL = """
create extension if not exists vector;

create table if not exists aptel_orders (
    id               text primary key,
    petition_no      text,
    cause_title      text,
    bench            text,
    date_of_decision date,
    date_uploaded    date,
    pdf_url          text,
    pdf_digest       text,
    pdf_fulltext     text,
    embedding        vector(384),
    scraped_at       timestamptz,
    fts tsvector generated always as (
        to_tsvector('english',
            coalesce(cause_title,'') || ' ' ||
            coalesce(petition_no,'') || ' ' ||
            coalesce(pdf_digest,'')  || ' ' ||
            coalesce(pdf_fulltext,''))
    ) stored
);

create index if not exists aptel_orders_embedding_idx
    on aptel_orders using hnsw (embedding vector_cosine_ops);
create index if not exists aptel_orders_fts_idx
    on aptel_orders using gin (fts);
"""

_st_model = None


# ================= EMBEDDING / SEMANTIC =================

def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (first run downloads ~80MB) …")
        _st_model = SentenceTransformer(MODEL_NAME)
    return _st_model


def embed_to_literal(text: str) -> str:
    """Embed one string -> pgvector literal '[...]'."""
    vec = _get_model().encode([text or "(empty)"], normalize_embeddings=True)[0]
    if len(vec) != EMBED_DIM:
        sys.exit(f"Embedding dim {len(vec)} != {EMBED_DIM}; wrong model?")
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def semantic_select(full_text: str, cap: int) -> str:
    """Relevance-rank paragraphs against the summary queries and keep the top
    ones up to `cap` chars. Falls back to a head slice if the model is absent."""
    chunks = [c.strip() for c in re.split(r"\n{2,}", full_text) if len(c.strip()) > 80]
    if not chunks:
        return full_text[:cap]
    if sum(len(c) for c in chunks) <= cap:
        return full_text[:cap]
    try:
        import numpy as np
        model = _get_model()
        chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)
        query_embs = model.encode(SUMMARY_QUERIES, show_progress_bar=False)
        chunk_unit = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
        scores = np.zeros(len(chunks))
        for q in query_embs:
            q_unit = q / (np.linalg.norm(q) + 1e-8)
            scores = np.maximum(scores, chunk_unit @ q_unit)
        # Keep highest-scoring chunks IN DOCUMENT ORDER up to the cap, so the
        # digest reads coherently and the most relevant chunks land first (which
        # is what MiniLM's 256-token window actually sees).
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        keep, used = set(), 0
        for i in order:
            if used + len(chunks[i]) > cap:
                continue
            keep.add(i); used += len(chunks[i])
            if used >= cap:
                break
        if not keep:
            keep = {int(order[0])}
        return "\n\n".join(chunks[i] for i in sorted(keep))
    except Exception as e:
        print(f"    [semantic fallback: {e}]")
        return full_text[:cap]


# ================= PDF EXTRACTION =================

def pdf_to_text(content: bytes, max_pages: int) -> str:
    """Complete text extraction. pymupdf4llm (table-aware Markdown) preferred;
    pdfplumber fallback. Returns '' if neither yields text."""
    # pymupdf4llm
    try:
        import fitz  # pymupdf
        import pymupdf4llm
        doc = fitz.open(stream=content, filetype="pdf")
        pages = list(range(min(len(doc), max_pages)))
        md = pymupdf4llm.to_markdown(doc, pages=pages, show_progress=False)
        if md and md.strip():
            return md.strip()
    except Exception as e:
        print(f"    [pymupdf4llm: {e}; trying pdfplumber]")
    # pdfplumber fallback
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception as e:
        print(f"    [pdfplumber: {e}]")
        return ""


# ================= SCRAPE (paginated, single year) =================

def make_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def _clean(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def _abs(href: str) -> str:
    href = href.strip()
    if href.startswith("http"):
        return href
    return BASE_URL.rstrip("/") + "/" + href.lstrip("/")


def parse_table(html: str) -> list:
    """Parse one listing page's judgements table. Mirrors the proven column
    layout in aptel_watcher.scrape_orders()."""
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for t in soup.find_all("table"):
        txt = t.get_text()
        if "APPEAL/PETITION" in txt or "CAUSE TITLE" in txt:
            table = t
            break
    if not table:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        link = cells[1].find("a", href=True)
        if not link or not link["href"].lower().endswith(".pdf"):
            continue
        pdf_url = _abs(link["href"])
        dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", _clean(cells[4]))
        rows.append({
            "id":               make_id(pdf_url),
            "petition_no":      _clean(cells[1]),
            "cause_title":      _clean(cells[2]),
            "bench":            _clean(cells[3]),
            "date_of_decision": dates[0] if dates else "",
            "date_uploaded":    dates[1] if len(dates) > 1 else (dates[0] if dates else ""),
            "pdf_url":          pdf_url,
        })
    return rows


def scrape_year(year: int, max_pages: int) -> list:
    """Page through the Drupal view for one year until no NEW rows appear."""
    session = requests.Session()
    session.headers.update(HEADERS)
    all_rows, seen, page = [], set(), 0
    while page < max_pages:
        url = f"{BASE_URL}{ORDERS_PATH}?field_judge_year_value={year}&page={page}"
        resp = session.get(url, timeout=40, verify=False)
        resp.raise_for_status()
        rows = parse_table(resp.text)
        fresh = [r for r in rows if r["id"] not in seen]
        if not fresh:
            # Out-of-range page numbers make Drupal repeat the last page; the
            # absence of new ids is our reliable end-of-pages signal.
            break
        for r in fresh:
            seen.add(r["id"])
        all_rows.extend(fresh)
        print(f"  page {page}: +{len(fresh)} (total {len(all_rows)})")
        page += 1
        time.sleep(0.5)
    return all_rows


# ================= DB =================

def parse_ddmmyyyy(s: str):
    s = (s or "").strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def db_connect():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL to your Supabase transaction-pooler string.")
    return psycopg2.connect(url)


def init_db():
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
        print("aptel_orders table + indexes ready.")
    finally:
        conn.close()


def existing_ids(conn) -> set:
    with conn.cursor() as cur:
        try:
            cur.execute("select id from aptel_orders")
            return {r[0] for r in cur.fetchall()}
        except Exception:
            conn.rollback()   # table not created yet
            return set()


UPSERT = """
    insert into aptel_orders
        (id, petition_no, cause_title, bench, date_of_decision, date_uploaded,
         pdf_url, pdf_digest, pdf_fulltext, embedding, scraped_at)
    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)
    on conflict (id) do update set
        petition_no      = excluded.petition_no,
        cause_title      = excluded.cause_title,
        bench            = excluded.bench,
        date_of_decision = excluded.date_of_decision,
        date_uploaded    = excluded.date_uploaded,
        pdf_url          = excluded.pdf_url,
        pdf_digest       = excluded.pdf_digest,
        pdf_fulltext     = coalesce(excluded.pdf_fulltext, aptel_orders.pdf_fulltext),
        embedding        = excluded.embedding,
        scraped_at       = excluded.scraped_at;
"""


# ================= MAIN =================

def main():
    ap = argparse.ArgumentParser(description="Cold-load a year of APTEL orders into Supabase.")
    ap.add_argument("--year", type=int, default=datetime.now().year)
    ap.add_argument("--dry-run", action="store_true",
                    help="Scrape the listing and report; download nothing, write nothing.")
    ap.add_argument("--init-db", action="store_true", help="Create table + indexes, then exit.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N new orders (0 = all).")
    ap.add_argument("--max-pages", type=int, default=50, help="Pager safety cap.")
    ap.add_argument("--max-pdf-pages", type=int, default=200,
                    help="Per-PDF page cap (APTEL judgments run long).")
    ap.add_argument("--request-delay", type=float, default=1.0, help="Seconds between PDF fetches.")
    ap.add_argument("--batch", type=int, default=20, help="Rows per commit.")
    ap.add_argument("--fulltext-dir", default="aptel/fulltext")
    args = ap.parse_args()

    if args.init_db:
        init_db()
        return

    print(f"Scraping APTEL judgements for {args.year} …")
    listing = scrape_year(args.year, args.max_pages)
    print(f"  {len(listing)} orders listed for {args.year}")

    # Resumability: skip what's already loaded.
    already = set()
    if not args.dry_run and os.environ.get("DATABASE_URL"):
        conn = db_connect()
        already = existing_ids(conn)
        conn.close()
        print(f"  {len(already)} already in aptel_orders; {len(listing) - len([r for r in listing if r['id'] in already])} candidates remain")

    todo = [r for r in listing if r["id"] not in already]
    if args.limit:
        todo = todo[:args.limit]

    if args.dry_run:
        dates = sorted(d for d in (parse_ddmmyyyy(r["date_of_decision"]) for r in listing) if d)
        print("\n--- DRY RUN ---")
        print(f"  year:            {args.year}")
        print(f"  listed:          {len(listing)}")
        print(f"  already in DB:   {len(already)}")
        print(f"  would download:  {len(todo)}")
        if dates:
            print(f"  decision dates:  {dates[0]} … {dates[-1]}")
        print("\n  first few:")
        for r in todo[:10]:
            print(f"    {r['date_of_decision']:>10}  {r['petition_no'][:70]}")
        print("\nNo PDFs fetched, no writes. Re-run without --dry-run to load.")
        return

    if not todo:
        print("Nothing to load. Done.")
        return

    os.makedirs(args.fulltext_dir, exist_ok=True)
    conn = db_connect()
    session = requests.Session()
    session.headers.update(HEADERS)
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
        for i, r in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {r['petition_no'][:70]}")
            try:
                resp = session.get(r["pdf_url"], timeout=60, verify=False)
                resp.raise_for_status()
            except Exception as e:
                print(f"    download failed, skipping (will retry next run): {e}")
                skipped += 1
                continue

            full_text = pdf_to_text(resp.content, args.max_pdf_pages)
            if not full_text:
                print("    no extractable text, skipping")
                skipped += 1
                continue

            # Store the COMPLETE text on disk (audit/grep) and in the column.
            with open(os.path.join(args.fulltext_dir, f"{r['id']}.md"), "w", encoding="utf-8") as f:
                f.write(full_text)

            digest = semantic_select(full_text[:HARD_CAP], DIGEST_CHARS)
            embedding = embed_to_literal(digest)

            batch.append((
                r["id"], r["petition_no"], r["cause_title"], r["bench"],
                parse_ddmmyyyy(r["date_of_decision"]), parse_ddmmyyyy(r["date_uploaded"]),
                r["pdf_url"], digest, full_text, embedding,
                datetime.now(timezone.utc).isoformat(),
            ))
            loaded += 1
            if len(batch) >= args.batch:
                flush()
                print(f"    … committed (running total {loaded})")
            if args.request_delay:
                time.sleep(args.request_delay)
        flush()
    finally:
        flush()
        conn.close()

    print(f"\nDone. Loaded {loaded}, skipped {skipped} (deferred to a re-run).")


if __name__ == "__main__":
    main()
