#!/usr/bin/env python3
"""
build_appeal_links.py — materialize the CERC -> APTEL appeal linkage.

Why this exists
---------------
The Atlas appeal badges and the set-aside tracker read from a table,
`aptel_cerc_links`. This script populates it. Without it, every CERC order shows
appeal posture 'unknown' (correctly — we haven't checked). With it, orders that
APTEL actually heard get a real status, and the set-aside tracker lights up.

It ports the same idea as the MCP `get_aptel_appeals` tool: a CERC order that was
appealed is identified by its CERC petition number appearing in the body of an
APTEL judgment. Instead of doing that ILIKE at query time, we do it once nightly
and store the result.

Matching (deliberately conservative)
------------------------------------
For each CERC petition_no we build a NORMALISED needle (digits + type token,
e.g. '235/MP/2021' -> '235/mp/2021', tolerant of spacing/slash variants) and
look for it in aptel_orders.pdf_fulltext (falling back to pdf_digest). A CERC
petition number is distinctive enough that a fulltext hit is a strong signal —
but it is still a CANDIDATE, not proof, so:
  * we store the matched APTEL id + a snippet for human verification,
  * disposition is left NULL unless we can read it cheaply from the judgment,
  * the UI labels the whole tracker "candidate linkage — verify against original".

Disposition (optional, best-effort)
-----------------------------------
If --read-disposition is set, we scan a window near the end of the APTEL text for
operative verbs (set aside / dismissed / allowed / remanded). This is heuristic
and frequently wrong on partly-allowed orders; leave it off for a purely factual
linkage and let the human read disposition from the order.

Table
-----
  aptel_cerc_links(
    cerc_petition_no text,
    cerc_doc_id      text,
    aptel_doc_id     text,
    disposition      text,          -- NULL unless --read-disposition found one
    match_snippet    text,          -- context around the hit, for verification
    matched_at       timestamptz default now(),
    primary key (cerc_petition_no, aptel_doc_id)
  )

Usage
-----
  python build_appeal_links.py                     # rebuild linkage
  python build_appeal_links.py --read-disposition  # also guess disposition
  python build_appeal_links.py --limit 200         # test on a slice
  DATABASE_URL=... python build_appeal_links.py

Run it in the nightly chain BEFORE export_atlas.py so the exporter sees fresh
links. Idempotent: it truncates and rebuilds (the match is cheap relative to a
full re-tag).
"""
import argparse
import os
import re
import sys

import psycopg2
import psycopg2.extras


def log(m):
    print(m, file=sys.stderr, flush=True)


DDL = """
create table if not exists aptel_cerc_links (
    cerc_petition_no text,
    cerc_doc_id      text,
    aptel_doc_id     text,
    disposition      text,
    match_snippet    text,
    matched_at       timestamptz default now(),
    primary key (cerc_petition_no, aptel_doc_id)
);
create index if not exists aptel_cerc_links_cerc_no
    on aptel_cerc_links (cerc_petition_no);
"""

# operative-language windows for optional disposition reading
DISP_PATTERNS = [
    ("set_aside", re.compile(r"\bset\s+aside\b", re.I)),
    ("remanded",  re.compile(r"\bremand(?:ed)?\b", re.I)),
    ("allowed",   re.compile(r"\bappeal(?:s)?\s+(?:is|are)\s+allowed\b", re.I)),
    ("dismissed", re.compile(r"\bappeal(?:s)?\s+(?:is|are)\s+dismissed\b", re.I)),
]


def norm_petition(pno):
    """235 / MP / 2021 -> canonical '235/mp/2021'; also return a tolerant regex."""
    if not pno:
        return None, None
    s = re.sub(r"\s+", "", str(pno)).lower()
    m = re.match(r"(\d+)/([a-z]{2,4})/(\d{4})", s)
    if not m:
        return s, None
    num, typ, yr = m.groups()
    canon = f"{num}/{typ}/{yr}"
    # tolerant: allow spaces or nothing around the slashes in the APTEL text
    rx = re.compile(rf"{num}\s*/\s*{typ}\s*/\s*{yr}", re.I)
    return canon, rx


def connect():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log("FATAL: DATABASE_URL not set"); sys.exit(2)
    return psycopg2.connect(dsn)


def read_disposition(text):
    if not text:
        return None
    tail = text[-4000:]  # operative part is near the end
    for name, rx in DISP_PATTERNS:
        if rx.search(tail):
            return name
    return None


def snippet(text, rx, width=160):
    m = rx.search(text or "")
    if not m:
        return None
    a = max(0, m.start() - width // 2)
    b = min(len(text), m.end() + width // 2)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-disposition", action="store_true",
                    help="heuristically guess disposition from APTEL text (noisy)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only scan the first N CERC orders (testing)")
    args = ap.parse_args()

    conn = connect()
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()

    # Pull APTEL corpus once into memory (id, text). Corpus is a few thousand
    # rows; fulltext is large but a single pass is fine on the runner.
    cur.execute("SELECT id, coalesce(pdf_fulltext, pdf_digest, '') FROM aptel_orders")
    aptel = cur.fetchall()
    log(f"aptel corpus: {len(aptel)} judgments loaded")

    # CERC petition numbers to look for.
    q = "SELECT id, petition_no FROM cerc_orders WHERE petition_no IS NOT NULL"
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    cur.execute(q)
    cerc = cur.fetchall()
    log(f"cerc orders to match: {len(cerc)}")

    rows = []
    matched_orders = 0
    for cerc_id, pno in cerc:
        canon, rx = norm_petition(pno)
        if not rx:
            continue
        hits = []
        for aptel_id, atext in aptel:
            if not atext:
                continue
            # cheap prefilter: the bare number must appear before running regex
            if canon.split("/")[0] not in atext:
                continue
            if rx.search(atext):
                disp = read_disposition(atext) if args.read_disposition else None
                hits.append((aptel_id, disp, snippet(atext, rx)))
        if hits:
            matched_orders += 1
            for aptel_id, disp, snip in hits:
                rows.append((canon, cerc_id, aptel_id, disp, snip))

    cur.execute("TRUNCATE aptel_cerc_links")
    if rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO aptel_cerc_links "
            "(cerc_petition_no, cerc_doc_id, aptel_doc_id, disposition, match_snippet) "
            "VALUES %s ON CONFLICT (cerc_petition_no, aptel_doc_id) DO NOTHING",
            rows,
        )
    conn.commit()
    log(f"linkage built: {len(rows)} links across {matched_orders} CERC orders "
        f"({'with' if args.read_disposition else 'without'} disposition reading)")
    cur.close(); conn.close()


if __name__ == "__main__":
    main()
