#!/usr/bin/env python3
"""
link_news.py — Tier A: precise, sparse entity+date candidate links between news
items and CERC/APTEL orders.

This is deliberately HIGH-PRECISION / LOW-RECALL. The two corpora barely share
vocabulary (orders speak petition numbers + formal party names; trade press
speaks project names + MW + brands), so we only assert a candidate link when a
strong, checkable signal co-occurs. Everything written here is a LEAD TO VERIFY,
surfaced in the UI as such — never an asserted relationship.

Three matchers (a news item can match by more than one; each is its own row):
  petition_no  — the news text quotes a CERC petition number that an order has.
                 Strongest signal.
  party+date   — a normalized party/org name shared between the news item and an
                 order, AND the order date within +/- `window` days of the item.
                 The date gate is what keeps common names (NTPC, Adani) from
                 producing spurious links across unrelated years.
  forum_mention— the item explicitly names CERC/APTEL and shares a party with an
                 order (weaker; only emitted when party+date didn't already fire).

Tier B (issue-level) links need NO table — they're a join on doc_tags subject
codes. This script only does Tier A.

Usage
-----
  python link_news.py                    # link all news items
  python link_news.py --window 45        # widen the date gate (default 30)
  python link_news.py --limit 200
  DATABASE_URL=... python link_news.py

Idempotent: truncates and rebuilds news_order_links.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import psycopg2
import psycopg2.extras


def log(m):
    print(m, file=sys.stderr, flush=True)


# ---- entity normalization ---------------------------------------------------

# Corporate suffixes / noise to strip so "NTPC Ltd." == "NTPC Limited" == "NTPC".
_SUFFIX = re.compile(
    r"\b(ltd|limited|pvt|private|corp|corporation|co|company|inc|"
    r"power|energy|india|renewables?|solar|wind|electricity|"
    r"distribution|transmission|generation|discom|genco|transco)\b",
    re.I,
)
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

# Known multi-word org anchors worth matching as units (extend over time).
ORG_ANCHORS = [
    "ntpc", "nhpc", "sjvn", "pgcil", "power grid", "adani", "tata power",
    "torrent power", "jsw", "reliance", "renew", "greenko", "azure power",
    "acme", "vedanta", "hindalco", "sterlite", "bses", "cesc", "gmr",
    "damodar valley", "dvc", "sembcorp", "avaada", "o2 power",
    "discom", "gujarat urja", "mseb", "mahagenco", "tangedco", "pspcl",
]


def norm_name(s):
    """Lowercase, strip suffixes/punct, collapse ws. For token-set comparison."""
    if not s:
        return ""
    s = _NONWORD.sub(" ", s.lower())
    s = _SUFFIX.sub(" ", s)
    return _WS.sub(" ", s).strip()


def anchors_in(text):
    """Set of known org anchors present in a text (normalized substring hit)."""
    t = " " + norm_name(text) + " "
    found = set()
    for a in ORG_ANCHORS:
        if f" {a} " in t or t.strip().startswith(a) or (" " + a) in t:
            found.add(a)
    return found


PET_RX = re.compile(r"\b(\d+)\s*/\s*([A-Za-z]{2,4})\s*/\s*(\d{4})\b")


def petition_numbers_in(text):
    out = set()
    for m in PET_RX.finditer(text or ""):
        out.add(f"{m.group(1)}/{m.group(2).lower()}/{m.group(3)}")
    return out


def norm_petition(pno):
    if not pno:
        return None
    s = re.sub(r"\s+", "", str(pno)).lower()
    m = re.match(r"(\d+)/([a-z]{2,4})/(\d{4})", s)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else s


def mentions_forum(text):
    t = (text or "").lower()
    forums = set()
    if "cerc" in t or "central electricity regulatory" in t:
        forums.add("CERC")
    if "aptel" in t or "appellate tribunal" in t:
        forums.add("APTEL")
    return forums


# ---- db ---------------------------------------------------------------------

def connect():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log("FATAL: DATABASE_URL not set"); sys.exit(2)
    return psycopg2.connect(dsn)


def fetch_orders(cur):
    """Minimal order rows for matching: (forum, id, petition_no, party_text, date)."""
    orders = []
    cur.execute("SELECT id, petition_no, subject, date_order FROM cerc_orders")
    for oid, pno, subj, d in cur.fetchall():
        orders.append(("CERC", str(oid), norm_petition(pno), subj or "", d))
    cur.execute("SELECT id, petition_no, cause_title, date_of_decision FROM aptel_orders")
    for oid, pno, ctitle, d in cur.fetchall():
        orders.append(("APTEL", str(oid), norm_petition(pno), ctitle or "", d))
    return orders


def fetch_news(cur, limit):
    q = ("SELECT id, title, coalesce(summary,''), coalesce(fulltext,''), published "
         "FROM news_items")
    if limit:
        q += f" LIMIT {int(limit)}"
    cur.execute(q)
    return cur.fetchall()


# ---- matching ---------------------------------------------------------------

def daydiff(a, b):
    if not a or not b:
        return None
    return abs((a - b).days)


def build_links(orders, news, window):
    # index orders by petition number and by anchor for cheap lookup
    by_pet = defaultdict(list)
    by_anchor = defaultdict(list)
    order_anchor = {}
    for forum, oid, pno, party_text, odate in orders:
        if pno:
            by_pet[pno].append((forum, oid, odate))
        ancs = anchors_in(party_text)
        order_anchor[(forum, oid)] = (ancs, odate)
        for a in ancs:
            by_anchor[a].append((forum, oid, odate))

    rows = []
    for nid, title, summary, fulltext, ndate in news:
        text = f"{title} {summary} {fulltext}"
        # 1) petition-number match (strongest)
        for pno in petition_numbers_in(text):
            for forum, oid, odate in by_pet.get(pno, []):
                rows.append((nid, forum, oid, "petition_no", 0.95,
                             f"quoted {pno}"))
        # 2) party + date proximity
        n_anchors = anchors_in(text)
        n_forums = mentions_forum(text)
        matched_pairs = set()
        for a in n_anchors:
            for forum, oid, odate in by_anchor.get(a, []):
                dd = daydiff(ndate, odate)
                if dd is not None and dd <= window:
                    score = round(0.6 + 0.3 * (1 - dd / window), 3)
                    rows.append((nid, forum, oid, "party+date", score,
                                 f"{a} within {dd}d"))
                    matched_pairs.add((forum, oid))
        # 3) forum mention + shared party (only if party+date didn't fire it)
        if n_forums:
            for a in n_anchors:
                for forum, oid, odate in by_anchor.get(a, []):
                    if forum in n_forums and (forum, oid) not in matched_pairs:
                        rows.append((nid, forum, oid, "forum_mention", 0.4,
                                     f"{a} + {forum} named"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30,
                    help="party+date match window in days")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    conn = connect()
    cur = conn.cursor()
    orders = fetch_orders(cur)
    news = fetch_news(cur, args.limit)
    log(f"orders: {len(orders)}  news: {len(news)}  window: {args.window}d")

    rows = build_links(orders, news, args.window)
    # dedup identical (news,forum,order,method) keeping best score
    best = {}
    for nid, forum, oid, method, score, ev in rows:
        k = (nid, forum, oid, method)
        if k not in best or score > best[k][4]:
            best[k] = (nid, forum, oid, method, score, ev)
    final = list(best.values())

    cur.execute("TRUNCATE news_order_links")
    if final:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO news_order_links "
            "(news_id, forum, order_id, method, score, evidence) VALUES %s "
            "ON CONFLICT DO NOTHING",
            final,
        )
    conn.commit()
    n_items = len({r[0] for r in final})
    by_method = defaultdict(int)
    for r in final:
        by_method[r[3]] += 1
    log(f"links: {len(final)} across {n_items} news items  {dict(by_method)}")
    cur.close(); conn.close()


if __name__ == "__main__":
    main()
