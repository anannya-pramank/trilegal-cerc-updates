#!/usr/bin/env python3
"""
export_atlas.py — publish the static JSON the Atlas site reads.

Reads the SAME Supabase that scrape.yml / tag.yml populate and writes flat JSON
files into docs/data/. No secret ever reaches the browser: this runs in Actions,
the browser only fetches the committed JSON.

Design rules
------------
* Tagging is PARTIAL (all APTEL tagged, some CERC pending). Nothing here assumes
  a document is tagged. An untagged order still appears in the index with empty
  facet arrays; coverage grows underneath the site as tag.yml catches up.
* Sharded, not one blob:
    index.json            — one light row per order (id, forum, no, date, title,
                            party, facet-tag CODES, appeal status). Drives all
                            browsing + client-side faceting.
    orders/<id>.json      — full detail for one order (summary, tags w/ names,
                            citations, appeal chain, related news). Lazy-loaded.
    taxonomy.json         — facet tree (code -> name/parent/def) for rendering
                            filter rails and tag labels.
    meta.json             — coverage stats + build timestamp (honest denominator).
    agg/*.json            — precomputed aggregates (dashboard, set-aside, issues).
* Idempotent + deterministic: sorted keys, stable ordering, so git diffs are
  meaningful and re-runs only change what actually changed.

Usage
-----
  python export_atlas.py                 # full export, default out dir docs/data
  python export_atlas.py --out docs/data --shards-only   # skip aggregates
  DATABASE_URL=... python export_atlas.py

The taxonomy facet CSVs (facet_1_subject_matter.csv, ...) are read from --tax-dir
(default: taxonomy/) so tag labels resolve to human names.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import psycopg2
import psycopg2.extras

# ---- config -----------------------------------------------------------------

CERC_TABLE = "cerc_orders"
APTEL_TABLE = "aptel_orders"

# petition-number suffix -> label (deterministic, mirrors the tagger regex).
PET_TYPE = {
    "MP": "Petition (Miscellaneous)", "TT": "Transmission Tariff",
    "GT": "Generation Tariff", "RP": "Review Petition", "SM": "Suo Motu",
    "RC": "Regulatory Compliance", "AT": "Adoption of Tariff (s.63)",
    "TDL": "Trans. Deviation / Licence", "FP": "Fee Petition",
    "RA": "Review Application",
}

# facets that carry a "disposition/outcome" meaning, for the set-aside view.
DISPOSITION_FACET = "disposition"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---- taxonomy ---------------------------------------------------------------

def load_taxonomy(tax_dir: Path):
    """Flatten the facet CSVs into {code: {name, parent, facet, definition}}."""
    nodes = {}
    if not tax_dir.exists():
        log(f"WARN: taxonomy dir {tax_dir} missing; tags will show codes only")
        return nodes
    for csv_path in sorted(tax_dir.glob("facet_*.csv")):
        facet = csv_path.stem  # e.g. facet_1_subject_matter
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("Code") or "").strip()
                if not code:
                    continue
                nodes[code] = {
                    "name": (row.get("Name") or code).strip(),
                    "parent": (row.get("Parent") or "").strip() or None,
                    "facet": facet,
                    "definition": (row.get("Definition and exclusions") or "").strip(),
                }
    log(f"taxonomy: {len(nodes)} nodes from {tax_dir}")
    return nodes


# ---- db ---------------------------------------------------------------------

def connect():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log("FATAL: DATABASE_URL not set")
        sys.exit(2)
    return psycopg2.connect(dsn)


def fetch_orders(cur, table, forum):
    """One row per order. Column names kept defensive: we SELECT * and pick."""
    cur.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    out = []
    for rec in cur.fetchall():
        row = dict(zip(cols, rec))
        out.append(_normalize_order(row, forum))
    log(f"{table}: {len(out)} orders")
    return out


def _first(row, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def _normalize_order(row, forum):
    """Map heterogeneous CERC/APTEL columns to one shape. Missing -> None."""
    oid = _first(row, "id", "doc_id", "hash", "content_hash")
    petition_no = _first(row, "petition_no", "case_no", "case_number", "diary_no")
    date = _first(row, "order_date", "date", "judgment_date", "pronounced_on")
    title = _first(row, "title", "cause_title", "subject")
    party = _first(row, "petitioner", "appellant", "party")
    summary = _first(row, "summary", "pdf_digest")
    url = _first(row, "pdf_url", "url", "source_url", "link")
    return {
        "id": str(oid) if oid is not None else None,
        "forum": forum,
        "petition_no": petition_no,
        "date": _iso_date(date),
        "title": _clean(title),
        "party": _clean(party),
        "pet_type": _pet_type(petition_no),
        "summary": _clean(summary),
        "url": url,
    }


def fetch_tags(cur):
    """doc_tags -> {(source, doc_id): [ {facet, code, confidence, method} ]}."""
    try:
        cur.execute(
            "SELECT source, doc_id, facet, node_code, confidence, method "
            "FROM doc_tags"
        )
    except psycopg2.Error:
        log("WARN: doc_tags not queryable; exporting with zero tags")
        return {}
    tags = defaultdict(list)
    for source, doc_id, facet, code, conf, method in cur.fetchall():
        tags[(source, str(doc_id))].append({
            "facet": facet, "code": code,
            "confidence": float(conf) if conf is not None else None,
            "method": method,
        })
    log(f"doc_tags: {sum(len(v) for v in tags.values())} tags "
        f"over {len(tags)} documents")
    return tags


def fetch_appeal_links(cur):
    """CERC petition_no -> APTEL appeal rows, best-effort from existing linkage."""
    links = defaultdict(list)
    try:
        cur.execute(
            "SELECT cerc_petition_no, aptel_doc_id, disposition "
            "FROM aptel_cerc_links"
        )
        for cerc_no, aptel_id, disp in cur.fetchall():
            links[cerc_no].append({"aptel_id": str(aptel_id), "disposition": disp})
    except psycopg2.Error:
        log("WARN: aptel_cerc_links absent; appeal badges will be 'unknown'")
    return links


# ---- helpers ----------------------------------------------------------------

_WS = re.compile(r"\s+")
_ARTIFACT = re.compile(r"<!--.*?-->|<mark>|</mark>|<[^>]+>")


def _clean(s):
    if not s:
        return None
    s = _ARTIFACT.sub(" ", str(s))
    s = s.replace("…", " ").replace("..........", " ")
    return _WS.sub(" ", s).strip() or None


def _iso_date(d):
    if d is None:
        return None
    if isinstance(d, (dt.date, dt.datetime)):
        return d.strftime("%Y-%m-%d")
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", str(d))
    if m:
        y, mo, da = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(da):02d}"
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", str(d))
    if m:
        da, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(da):02d}"
    return None


def _pet_type(pet_no):
    if not pet_no:
        return None
    m = re.search(r"/([A-Z]{2,3})/", str(pet_no))
    return m.group(1) if m and m.group(1) in PET_TYPE else None


def _year(iso):
    return iso[:4] if iso else None


# ---- assembly ---------------------------------------------------------------

def build(out_dir: Path, tax_dir: Path, shards_only=False):
    taxonomy = load_taxonomy(tax_dir)
    conn = connect()
    cur = conn.cursor()

    orders = (fetch_orders(cur, APTEL_TABLE, "APTEL")
              + fetch_orders(cur, CERC_TABLE, "CERC"))
    tags = fetch_tags(cur)
    appeal_links = fetch_appeal_links(cur)
    cur.close(); conn.close()

    orders = [o for o in orders if o["id"]]
    orders.sort(key=lambda o: (o["date"] or "0000", o["forum"], o["id"]))

    orders_dir = out_dir / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)

    index = []
    tagged_count = 0
    facet_counts = defaultdict(Counter)

    for o in orders:
        key = (o["forum"], o["id"])
        otags = tags.get(key, [])
        if otags:
            tagged_count += 1
        codes = sorted({t["code"] for t in otags})
        for t in otags:
            facet_counts[t["facet"]][t["code"]] += 1

        appeal = _appeal_status(o, appeal_links)

        # light index row — everything the browser needs to facet/sort/list.
        index.append({
            "id": o["id"], "forum": o["forum"], "no": o["petition_no"],
            "date": o["date"], "title": o["title"], "party": o["party"],
            "pt": o["pet_type"], "tags": codes, "appeal": appeal["status"],
        })

        # per-order detail — lazy-loaded on click.
        detail = dict(o)
        detail["tags"] = [
            {**t, "name": taxonomy.get(t["code"], {}).get("name", t["code"]),
             "facet_of": taxonomy.get(t["code"], {}).get("facet")}
            for t in sorted(otags, key=lambda x: (x["facet"], -(x["confidence"] or 0)))
        ]
        detail["appeal"] = appeal
        _write_json(orders_dir / f"{_safe(o['id'])}.json", detail)

    _write_json(out_dir / "index.json", index)
    _write_json(out_dir / "taxonomy.json", taxonomy)

    total = len(orders)
    cerc_total = sum(1 for o in orders if o["forum"] == "CERC")
    cerc_tagged = sum(1 for o in orders if o["forum"] == "CERC"
                      and tags.get((o["forum"], o["id"])))
    aptel_total = total - cerc_total
    aptel_tagged = tagged_count - cerc_tagged
    meta = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "orders_total": total,
        "orders_tagged": tagged_count,
        "coverage": {
            "APTEL": {"total": aptel_total, "tagged": aptel_tagged},
            "CERC": {"total": cerc_total, "tagged": cerc_tagged},
        },
    }
    _write_json(out_dir / "meta.json", meta)
    log(f"index: {total} orders, {tagged_count} tagged "
        f"(CERC {cerc_tagged}/{cerc_total}, APTEL {aptel_tagged}/{aptel_total})")

    if not shards_only:
        _write_aggregates(out_dir, orders, tags, taxonomy, appeal_links)
    log(f"done -> {out_dir}")


def _appeal_status(order, appeal_links):
    if order["forum"] != "CERC":
        return {"status": "n/a", "chain": []}
    chain = appeal_links.get(order["petition_no"], [])
    if not chain:
        return {"status": "unappealed", "chain": []}
    disps = {(c.get("disposition") or "").lower() for c in chain}
    if any("set aside" in d or "allowed" in d for d in disps):
        status = "set_aside"
    elif any("dismiss" in d or "affirm" in d for d in disps):
        status = "affirmed"
    elif any("remand" in d for d in disps):
        status = "remanded"
    else:
        status = "appealed_pending"
    return {"status": status, "chain": chain}


def _write_aggregates(out_dir, orders, tags, taxonomy, appeal_links):
    agg_dir = out_dir / "agg"
    agg_dir.mkdir(exist_ok=True)

    # dashboard: counts per year x forum
    by_year = defaultdict(lambda: {"CERC": 0, "APTEL": 0})
    for o in orders:
        y = _year(o["date"])
        if y:
            by_year[y][o["forum"]] += 1
    dashboard = [{"year": y, **by_year[y]} for y in sorted(by_year)]
    _write_json(agg_dir / "dashboard.json", dashboard)

    # set-aside tracker: CERC orders disturbed on appeal
    set_aside = []
    for o in orders:
        if o["forum"] != "CERC":
            continue
        st = _appeal_status(o, appeal_links)
        if st["status"] in ("set_aside", "remanded"):
            set_aside.append({
                "id": o["id"], "no": o["petition_no"], "date": o["date"],
                "title": o["title"], "status": st["status"], "chain": st["chain"],
            })
    _write_json(agg_dir / "set_aside.json", set_aside)

    # issue frequency: subject-matter facet only, with human names
    subj_counter = Counter()
    for (forum, oid), otags in tags.items():
        for t in otags:
            if (t.get("facet") or "").startswith("facet_1"):
                subj_counter[t["code"]] += 1
    issues = [
        {"code": c, "name": taxonomy.get(c, {}).get("name", c), "count": n}
        for c, n in subj_counter.most_common()
    ]
    _write_json(agg_dir / "issues.json", issues)
    log(f"aggregates: dashboard({len(dashboard)}), "
        f"set_aside({len(set_aside)}), issues({len(issues)})")


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def _write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--tax-dir", default="taxonomy")
    ap.add_argument("--shards-only", action="store_true")
    args = ap.parse_args()
    build(Path(args.out), Path(args.tax_dir), args.shards_only)


if __name__ == "__main__":
    main()
