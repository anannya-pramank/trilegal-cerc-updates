"""
backfill_summaries.py — populate the summary column for rows ALREADY in
Supabase. Runs entirely from your local machine: it only needs DATABASE_URL
and network access to Supabase, both of which the MCP servers already use.
The GitHub-hosted cold loaders are irrelevant here — the fulltext this reads
is in the database, not in the repo.

    pip requirements: psycopg2-binary, sentence-transformers, numpy
    (all already installed for cerc_mcp.py / aptel_mcp.py)

Usage (from the directory containing extractive_summary.py):

    export DATABASE_URL='postgresql://...:6543/postgres'   # same as MCP config
    python backfill_summaries.py --table cerc_orders --limit 50   # smoke test
    python backfill_summaries.py --table cerc_orders              # full pass
    python backfill_summaries.py --table aptel_orders

Idempotent and resumable: only rows with a missing/stale summary are touched
(keyset pagination on the text-hash id, batch commits). Interrupt any time;
rerun to continue. Bump SUMMARY_VERSION in extractive_summary.py to force
regeneration of the whole corpus later.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg2
from psycopg2.extras import execute_values

from extractive_summary import SUMMARY_VERSION, TABLES, build_digest


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def remaining_count(conn, table: str, text_col: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from {table} where {text_col} is not null "
            f"and length({text_col}) >= 200 "
            f"and (summary is null or summary_model is distinct from %s)",
            (SUMMARY_VERSION,),
        )
        return cur.fetchone()[0]


def run(table: str, batch: int, limit: int | None) -> None:
    if table not in TABLES:
        raise SystemExit(f"table must be one of {sorted(TABLES)}")
    text_col = TABLES[table]["text_col"]

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = False

    todo = remaining_count(conn, table, text_col)
    print(f"[{table}] rows needing summaries: {todo}", flush=True)
    if todo == 0:
        conn.close()
        return

    model = get_model()
    print("MiniLM loaded", flush=True)

    # id is the scraper's TEXT hash — keyset paginate with string comparison,
    # starting below any possible value.
    last_id = ""
    done = 0
    t0 = time.time()

    fetch_sql = f"""
        select id, {text_col}
        from {table}
        where id > %s
          and {text_col} is not null
          and length({text_col}) >= 200
          and (summary is null or summary_model is distinct from %s)
        order by id
        limit %s
    """

    while True:
        with conn.cursor() as cur:
            cur.execute(fetch_sql, (last_id, SUMMARY_VERSION, batch))
            rows = cur.fetchall()
        if not rows:
            break

        updates = []
        for oid, fulltext in rows:
            last_id = oid
            try:
                digest = build_digest(fulltext, model)
            except Exception as e:   # one bad order must not kill the run
                print(f"  ! id={oid} failed: {e}", file=sys.stderr, flush=True)
                continue
            if digest:
                updates.append((oid, digest))

        if updates:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    update {table} t
                    set summary = d.summary,
                        summary_model = d.model,
                        summarized_at = now()
                    from (values %s) as d(id, summary, model)
                    where t.id = d.id
                    """,
                    [(oid, digest, SUMMARY_VERSION) for oid, digest in updates],
                )
            conn.commit()

        done += len(rows)
        rate = done / max(time.time() - t0, 1e-6)
        eta_min = (todo - done) / max(rate, 1e-6) / 60
        print(f"  {done}/{todo}  ({rate:.1f} rows/s, ~{eta_min:.0f} min left)",
              flush=True)

        if limit and done >= limit:
            print("--limit reached; rerun without it to continue", flush=True)
            break

    conn.close()
    print(f"[{table}] done: {done} rows processed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=sorted(TABLES))
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows (smoke testing)")
    args = ap.parse_args()
    run(args.table, args.batch, args.limit)
