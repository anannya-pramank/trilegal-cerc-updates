#!/usr/bin/env python3
"""
cerc_mcp.py — MCP server over the CERC orders store in Supabase.

Exposes four tools to an MCP client (e.g. Claude Desktop, GitHub Copilot):

  search_orders  — semantic search over order digests, with optional
                   date-range and category filters.
  order_stats    — aggregate counts (by month / quarter / year / category)
                   for trend questions like "this year vs last".
  get_order      — full record for one order by id or petition number.
  list_recent    — the most recently posted orders (the "what's new" view).

Runs as an SSE HTTP server (for remote MCP clients like GitHub Copilot cloud agent).
The embedding model is the same MiniLM used by the loader, so query vectors
match the stored 384-dim vectors.

Setup:
    pip install "mcp[cli]" psycopg2-binary sentence-transformers uvicorn
    export DATABASE_URL="postgresql://postgres.<ref>:<pw>@<host>:6543/postgres"
    python cerc_mcp.py
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from mcp.server.fastmcp import FastMCP

MODEL_NAME = "all-MiniLM-L6-v2"   # must match the loader's model (384-dim)

mcp = FastMCP("cerc")

# Load the embedding model lazily on first use, NOT at import time.
# Loading at import delays the MCP handshake and can trip the client's
# startup timeout. _get_model() loads once, then reuses.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed(text: str) -> str:
    """Embed a query string and return it as a pgvector literal '[...]'."""
    vec = _get_model().encode([text], normalize_embeddings=True)[0]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


@contextmanager
def _db():
    """Open a short-lived connection to Supabase (transaction pooler)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.close()


# Reciprocal Rank Fusion constant. 60 is the standard default from the
# original RRF paper; it controls how quickly a result's contribution
# decays with rank. Larger = flatter (lower ranks still matter).
RRF_K = 60


@mcp.tool()
def search_orders(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    category: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Hybrid search over CERC orders: semantic (vector) + keyword
    (full-text) retrieval, fused with Reciprocal Rank Fusion.

    This combines meaning-based matching (good for paraphrased concepts,
    e.g. "tariff true-up") with exact-token matching (good for petition
    numbers, section numbers, party names, e.g. "256/TT/2025" or
    "Section 62"). A result ranked highly by EITHER method surfaces.

    Args:
        query: What to find — natural language, a petition/section
            number, or a mix (e.g. "Section 62 transmission tariff").
        date_from: Optional ISO date (YYYY-MM-DD); only orders with
            date_order on or after this are returned.
        date_to: Optional ISO date (YYYY-MM-DD); only orders with
            date_order on or before this are returned.
        category: Optional exact category filter (e.g. "Generation",
            "Licence", "Misc.").
        limit: Maximum number of orders to return (default 10).

    Returns the best matching orders, each with id, petition_no,
    subject, category, date_order, pdf_url, a digest snippet, and an
    rrf_score (higher = better; it's a fused rank score, not a
    similarity percentage).
    """
    qvec = _embed(query)

    # Filters apply identically to both halves of the search.
    filters = []
    filter_params: list = []
    if date_from:
        filters.append("date_order >= %s")
        filter_params.append(date_from)
    if date_to:
        filters.append("date_order <= %s")
        filter_params.append(date_to)
    if category:
        filters.append("category = %s")
        filter_params.append(category)
    extra = (" and " + " and ".join(filters)) if filters else ""

    # Pull a wider candidate pool from each arm than the final limit, so
    # fusion has room to promote items that one arm ranked mid-list.
    pool = max(limit * 4, 40)

    # Two ranked CTEs (vector, keyword), each numbered by its own rank,
    # then full-outer-joined and scored by RRF: 1/(k+rank) summed across
    # whichever arms found the row.
    sql = f"""
        with vec as (
            select id, row_number() over (order by embedding <=> %s::vector) as rnk
            from cerc_orders
            where embedding is not null{extra}
            order by embedding <=> %s::vector
            limit %s
        ),
        kw as (
            select id, row_number() over (
                       order by ts_rank_cd(fts, websearch_to_tsquery('english', %s)) desc
                   ) as rnk
            from cerc_orders
            where fts @@ websearch_to_tsquery('english', %s){extra}
            limit %s
        ),
        fused as (
            select coalesce(vec.id, kw.id) as id,
                   coalesce(1.0/(%s + vec.rnk), 0)
                 + coalesce(1.0/(%s + kw.rnk), 0) as rrf_score
            from vec full outer join kw on vec.id = kw.id
        )
        select c.id, c.petition_no, c.subject, c.category,
               c.date_order, c.pdf_url,
               left(c.pdf_digest, 600) as digest_snippet,
               f.rrf_score
        from fused f join cerc_orders c on c.id = f.id
        order by f.rrf_score desc
        limit %s
    """
    params = [
        qvec, qvec, *filter_params, pool,          # vec CTE
        query, query, *filter_params, pool,        # kw CTE
        RRF_K, RRF_K,                              # RRF constants
        limit,                                     # final limit
    ]

    with _db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(r) | {"date_order": str(r["date_order"]) if r["date_order"] else None}
            for r in rows]


@mcp.tool()
def order_stats(
    group_by: str = "month",
    date_from: str | None = None,
    date_to: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """
    Aggregate order counts for trend analysis.

    Args:
        group_by: One of "month", "quarter", "year", or "category".
        date_from: Optional ISO date lower bound on date_order.
        date_to: Optional ISO date upper bound on date_order.
        category: Optional category filter applied before grouping.

    Returns a list of {period/category, count} rows, ordered. Use this
    for questions like "how many orders this year vs last" (group_by
    "year") or "which categories are most active" (group_by "category").
    """
    valid = {"month", "quarter", "year", "category"}
    if group_by not in valid:
        raise ValueError(f"group_by must be one of {sorted(valid)}")

    filters = ["date_order is not null"]
    params: list = []
    if date_from:
        filters.append("date_order >= %s")
        params.append(date_from)
    if date_to:
        filters.append("date_order <= %s")
        params.append(date_to)
    if category:
        filters.append("category = %s")
        params.append(category)
    where = " and ".join(filters)

    if group_by == "category":
        sql = f"""
            select coalesce(category, '(uncategorised)') as bucket, count(*) as count
            from cerc_orders where {where}
            group by bucket order by count desc
        """
    else:
        sql = f"""
            select to_char(date_trunc('{group_by}', date_order), 'YYYY-MM-DD') as bucket,
                   count(*) as count
            from cerc_orders where {where}
            group by bucket order by bucket
        """

    with _db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def get_order(identifier: str) -> dict | None:
    """
    Fetch the full record for a single order.

    Args:
        identifier: Either the order id (the scraper's hash) or the
            petition number (e.g. "256/TT/2025").

    Returns the complete order including the full pdf_digest, or null if
    no match is found.
    """
    sql = """
        select id, petition_no, subject, category, date_order, date_posted,
               pdf_url, pdf_digest, scraped_at
        from cerc_orders
        where id = %s or petition_no = %s
        limit 1
    """
    with _db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (identifier, identifier))
            row = cur.fetchone()
    if not row:
        return None
    r = dict(row)
    r["date_order"] = str(r["date_order"]) if r["date_order"] else None
    r["date_posted"] = str(r["date_posted"]) if r["date_posted"] else None
    r["scraped_at"] = str(r["scraped_at"]) if r["scraped_at"] else None
    return r


@mcp.tool()
def list_recent(days: int = 30, limit: int = 20) -> list[dict]:
    """
    List the most recently posted orders.

    Args:
        days: Look-back window on date_posted (default 30).
        limit: Maximum number of orders to return (default 20).

    Returns recent orders ordered newest-first, each with id,
    petition_no, subject, category, date_order, date_posted, pdf_url.
    Use this for "what's new" / "anything recent" questions.
    """
    sql = """
        select id, petition_no, subject, category, date_order, date_posted, pdf_url
        from cerc_orders
        where date_posted >= (current_date - %s::int)
        order by date_posted desc, date_order desc
        limit %s
    """
    with _db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (days, limit))
            rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["date_order"] = str(d["date_order"]) if d["date_order"] else None
        d["date_posted"] = str(d["date_posted"]) if d["date_posted"] else None
        out.append(d)
    return out


if __name__ == "__main__":
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    port = int(os.environ.get("PORT", 8000))
    # streamable_http_app() exposes /mcp — preferred by VS Code 1.99+
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
