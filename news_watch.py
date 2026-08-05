#!/usr/bin/env python3
"""
news_watch.py — energy-regulation news ingester.

Reads news_sources.yaml, pulls each RSS feed, applies a two-track admission gate
(trusted-scoped OR keyword-matched), embeds survivors with the SAME MiniLM model
the taxonomy uses, and upserts into news_items. Mirrors the ESG scraper's shape:
declarative sources, deterministic relevance score, dedup by URL hash, tolerant
of dead feeds, cron-job.org-friendly.

The MiniLM embedding at ingest is the key reuse trick: it lets tag_news.py run
the identical shortlist->classify path as tag_documents.py (which shortlists on
stored pgvector embeddings), so news gets tagged into the same taxonomy with no
new classifier code.

Usage
-----
  python news_watch.py                      # pull all sources
  python news_watch.py --source mercom      # one source
  python news_watch.py --no-embed           # skip embedding (faster; tag later)
  python news_watch.py --dry-run            # parse + gate, print, don't write
  DATABASE_URL=... python news_watch.py

Deps: feedparser, sentence-transformers (already in repo via the tagger), pyyaml,
psycopg2-binary, pgvector.
"""
import argparse
import hashlib
import os
import re
import sys
import datetime as dt
from pathlib import Path

import yaml
import psycopg2
import psycopg2.extras


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # same family the taxonomy uses


def log(m):
    print(m, file=sys.stderr, flush=True)


# ---- config -----------------------------------------------------------------

def load_sources(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    for s in cfg["sources"]:
        s.setdefault("min_keyword_hits", defaults.get("min_keyword_hits", 1))
        s.setdefault("scoped", False)
        s["_keywords"] = [k.lower() for k in defaults.get("keywords", [])]
    return cfg["sources"]


# ---- fetch + parse ----------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(s):
    if not s:
        return None
    s = _TAG.sub(" ", str(s))
    s = (s.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
           .replace("&nbsp;", " ").replace("&#8217;", "'").replace("&#8216;", "'"))
    return _WS.sub(" ", s).strip() or None


def item_id(url):
    canon = re.sub(r"[#?].*$", "", (url or "").strip().lower())
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def parse_date(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return dt.date(t.tm_year, t.tm_mon, t.tm_mday).isoformat()
    return None


def relevance_and_gate(title, summary, source):
    """Deterministic keyword score + admission decision.
    Returns (admitted: bool, score: float, hits: int)."""
    hay = f"{title or ''} {summary or ''}".lower()
    hits = sum(1 for k in source["_keywords"] if k in hay)
    # score in 0..1: saturating on hits, small bonus for CERC/APTEL explicit
    score = min(1.0, hits / 5.0)
    if any(x in hay for x in ("cerc", "aptel", "appellate tribunal")):
        score = min(1.0, score + 0.3)
    admitted = source["scoped"] or hits >= source["min_keyword_hits"]
    return admitted, round(score, 3), hits


def pull_source(source, dry_run=False):
    import feedparser   # lazy: only needed for live RSS, not for importers reusing helpers
    rows = []
    seen = set()
    for feed_url in source["feeds"]:
        try:
            fp = feedparser.parse(feed_url)
        except Exception as e:                      # dead/blocked feed: skip
            log(f"  [{source['id']}] feed error {feed_url}: {e}")
            continue
        if fp.bozo and not fp.entries:
            log(f"  [{source['id']}] no entries from {feed_url}")
            continue
        for e in fp.entries:
            url = e.get("link")
            if not url:
                continue
            iid = item_id(url)
            if iid in seen:
                continue
            seen.add(iid)
            title = clean(e.get("title"))
            summary = clean(e.get("summary") or e.get("description"))
            if not title:
                continue
            admitted, score, hits = relevance_and_gate(title, summary, source)
            if not admitted:
                continue
            rows.append({
                "id": iid, "source": source["id"], "url": url, "title": title,
                "published": parse_date(e), "summary": summary,
                "relevance": score,
            })
    log(f"  [{source['id']}] admitted {len(rows)} items")
    return rows


# ---- embedding (reuse the taxonomy's MiniLM) --------------------------------

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        log(f"loading {MODEL_NAME} …")
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_rows(rows):
    if not rows:
        return
    model = get_model()
    texts = [f"{r['title']}. {r.get('summary') or ''}"[:1000] for r in rows]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for r, v in zip(rows, vecs):
        r["embedding"] = "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


# ---- db ---------------------------------------------------------------------

def connect():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log("FATAL: DATABASE_URL not set"); sys.exit(2)
    return psycopg2.connect(dsn)


def upsert(conn, rows, with_embed):
    if not rows:
        return 0
    # Dedup by id (last-wins). Postgres ON CONFLICT DO UPDATE raises
    # CardinalityViolation if one statement proposes the same id twice, which
    # happens when two source URLs normalize to the same id. Collapse here so
    # no caller can trigger it.
    deduped = {}
    for r in rows:
        deduped[r["id"]] = r
    rows = list(deduped.values())
    cols = ["id", "source", "url", "title", "published", "summary", "relevance"]
    if with_embed:
        cols.append("embedding")
    tmpl = "(" + ",".join(["%s"] * len(cols)) + ")"
    values = [[r.get(c) for c in cols] for r in rows]
    setcols = ",".join(f"{c}=excluded.{c}" for c in cols if c != "id")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"insert into news_items ({','.join(cols)}) values %s "
            f"on conflict (id) do update set {setcols}, fetched_at=now()",
            values, template=tmpl,
        )
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-file", default="news_sources.yaml")
    ap.add_argument("--source", help="only this source id")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sources = load_sources(args.sources_file)
    if args.source:
        sources = [s for s in sources if s["id"] == args.source]
        if not sources:
            log(f"no source '{args.source}'"); sys.exit(1)

    all_rows = []
    for s in sources:
        log(f"[{s['id']}] {s['name']}")
        all_rows.extend(pull_source(s, args.dry_run))

    log(f"total admitted: {len(all_rows)}")
    if args.dry_run:
        for r in all_rows[:40]:
            log(f"  {r['published']}  ({r['relevance']})  {r['title'][:90]}")
        log("dry-run: nothing written"); return

    with_embed = not args.no_embed
    if with_embed:
        embed_rows(all_rows)

    conn = connect()
    n = upsert(conn, all_rows, with_embed)
    conn.close()
    log(f"upserted {n} news items ({'with' if with_embed else 'no'} embeddings)")


if __name__ == "__main__":
    main()
