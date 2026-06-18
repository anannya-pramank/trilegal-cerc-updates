#!/usr/bin/env python3
"""
load_cerc.py
------------
Reads the CERC scraper's cerc_orders_new.json, embeds each order's
pdf_digest with a local MiniLM model (384-dim, no API key, nothing
leaves the machine), and upserts the rows into the Supabase
`cerc_orders` table.

Re-running is safe: rows are keyed on `id`, so existing orders are
updated in place rather than duplicated.

Usage (PowerShell):
    $env:DATABASE_URL = "postgresql://postgres.<ref>:<password>@<host>:6543/postgres"
    python load_cerc.py cerc_orders_new.json

The DATABASE_URL is the Supabase transaction-pooler string
(Connect -> Transaction pooler, port 6543). If the password contains
special characters like '@', URL-encode them (@ becomes %40).
"""

import json
import os
import sys
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, matches vector(384)
EMBED_DIM = 384


def pick_text(order: dict) -> str:
    """
    What we embed. pdf_digest is the CERC scraper's ranked extract — short
    and already distilled, so it's ideal for MiniLM. Fall back to subject,
    then petition_no, so we never embed an empty string.
    """
    digest = (order.get("pdf_digest") or "").strip()
    if digest:
        return digest
    subject = (order.get("subject") or "").strip()
    if subject:
        return subject
    return (order.get("petition_no") or "").strip() or "(no text)"


def parse_cerc_date(value):
    """
    CERC writes dates as DD.MM.YYYY (e.g. '15.06.2026'). Convert to a
    Python date so Postgres stores it correctly. Returns None if absent
    or unparseable (rather than crashing the whole load on one bad row).
    """
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None  # unrecognised format -> store NULL rather than fail


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python load_cerc.py path/to/cerc_orders_new.json")

    json_path = sys.argv[1]
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("Set DATABASE_URL to your Supabase transaction-pooler connection string.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Optional sidecar: cerc_orders_fulltext.json (gitignored) holds full
    # extracted text keyed by id. If present, load it so we can populate the
    # pdf_fulltext column. Absent -> full text simply stays null.
    fulltext_map = {}
    ft_path = os.path.join(os.path.dirname(json_path) or ".", "cerc_orders_fulltext.json")
    if os.path.exists(ft_path):
        try:
            with open(ft_path, "r", encoding="utf-8") as f:
                fulltext_map = json.load(f).get("fulltext", {})
            print(f"Loaded full text for {len(fulltext_map)} orders from sidecar.")
        except Exception as e:
            print(f"Warning: could not read fulltext sidecar ({e}); continuing without.")

    # CERC shape is {generated_at, count, items: [...]}. Also tolerate a bare list.
    if isinstance(data, dict):
        orders = data.get("items") or data.get("orders") or []
    else:
        orders = data
    if not orders:
        sys.exit("No orders found in the JSON file (looked for 'items').")

    print(f"Read {len(orders)} orders from {json_path}")

    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [pick_text(o) for o in orders]
    print("Embedding...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    rows = []
    for order, emb in zip(orders, embeddings):
        if len(emb) != EMBED_DIM:
            sys.exit(f"Embedding dim {len(emb)} != expected {EMBED_DIM}; check the model.")
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        oid = str(order.get("id") or "").strip()
        rows.append((
            oid,
            order.get("petition_no"),
            order.get("subject"),
            parse_cerc_date(order.get("date_order")),
            parse_cerc_date(order.get("date_posted")),
            order.get("category"),
            order.get("pdf_url"),
            order.get("pdf_digest"),
            fulltext_map.get(oid),     # full text from sidecar, or None
            vec_literal,
            order.get("scraped_at"),   # ISO timestamp, Postgres parses directly
        ))

    rows = [r for r in rows if r[0]]  # drop rows with no id
    print(f"Prepared {len(rows)} rows with valid ids.")

    sql = """
        insert into cerc_orders
            (id, petition_no, subject, date_order, date_posted,
             category, pdf_url, pdf_digest, pdf_fulltext, embedding, scraped_at)
        values %s
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

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            execute_values(
                cur, sql, rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)",
            )
        conn.commit()
    finally:
        conn.close()

    print(f"Done. Upserted {len(rows)} orders into cerc_orders.")


if __name__ == "__main__":
    main()
