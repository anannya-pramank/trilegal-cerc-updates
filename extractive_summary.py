"""
extractive_summary.py — API-free order digests for cerc_mcp.py / aptel_mcp.py.

Sits in the SAME directory as the two MCP server files. No new dependencies:
uses the psycopg2 + sentence-transformers already required by the servers,
and reuses the server's lazily-loaded MiniLM instance.

Schema-matched to the real tables:
    cerc_orders  (id text hash, pdf_fulltext, date_order,       ...)
    aptel_orders (id text hash, pdf_fulltext, date_of_decision, ...)

Migration (run once per table in Supabase SQL editor):

    alter table cerc_orders
      add column if not exists summary text,
      add column if not exists summary_model text,
      add column if not exists summarized_at timestamptz;

    alter table aptel_orders
      add column if not exists summary text,
      add column if not exists summary_model text,
      add column if not exists summarized_at timestamptz;
"""

from __future__ import annotations

import re
import numpy as np

SUMMARY_VERSION = "extractive-v6"

# Whitelisted per-table config — table names are NEVER interpolated from
# caller input, mirroring the _GREPPABLE pattern in the servers.
TABLES = {
    "cerc_orders":  {"text_col": "pdf_fulltext"},
    "aptel_orders": {"text_col": "pdf_fulltext"},
}

# ---------------------------------------------------------------------------
# 1. Paragraph splitting and boilerplate stripping
# ---------------------------------------------------------------------------

HEAD_NOISE = re.compile(
    r"^\s*(coram|in the matter of|and\s+in the matter of|petitioner|respondent"
    r"|appellant|versus|vs\.?|for the (petitioner|respondent|appellant)s?"
    r"|counsel|advocate|present:|parties present|date of (hearing|order|decision))",
    re.IGNORECASE,
)

BODY_START = re.compile(
    r"^\s*(ORDER|JUDGMENT|JUDGEMENT|DAILY ORDER)\s*$|^\s*1\s*[\.\)]\s+\S"
)

# Operative / high-signal language in CERC orders and APTEL judgements.
OPERATIVE = re.compile(
    r"(condon\w*|we direct|is (hereby )?(disposed|allowed|dismissed|rejected|remanded)"
    r"|hereby|held that|we hold|it is clarified|accordingly|liberty to"
    r"|true[- ]?up|carrying cost|prudence check|delay of \d+\s*(days|months)"
    r"|impugned order|set aside|in view of the above|summary of (our )?findings"
    r"|change in law|capital cost|annual fixed charge|tariff (is|shall))",
    re.IGNORECASE,
)

PARA_NUM = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s+")

# Cause-title respondent/petitioner entries: org name + address + PIN code.
# These are numbered like body paragraphs, so PARA_NUM alone can't exclude
# them — detect by address vocabulary and 6-digit PIN.
ADDRESS_LIKE = re.compile(
    r"\b\d{3}\s?\d{3}\b|"                           # PIN code, incl. "500 063"
    r"\b(Bhawan|Bhavan|Complex|Marg|Nagar|Sector[- ]\d+|Vidyut|Shakti"
    r"|House|Road,|Place,|District)\b",
    re.IGNORECASE,
)

# Bare cause-title entries: "4. Central Power Distribution Company of A.P
# Limited," — numbered, short, org suffix near the end, trailing comma or
# nothing after it. Tolerates abbreviation dots (A.P, M.P).
ORG_LINE = re.compile(
    r"^\s*\d{1,2}\s*[\.\)]\s+[A-Z].{5,140}?"
    r"\b(Limited|Ltd\.?|Corporation|Company|Board|Nigam|Utility|Utilities)\b"
    r"[^a-z]{0,10},?\s*$"
)


def _is_address(text: str) -> bool:
    if ORG_LINE.match(text):
        return True
    return (len(text) < 350
            and ADDRESS_LIKE.search(text) is not None
            and OPERATIVE.search(text) is None)


# pymupdf4llm renders quoted regulations as markdown emphasis/headings and
# tables as pipe rows — sometimes behind a leading list bullet ("- _a) ..._").
# Quoted law is not the Commission's finding; tables don't survive digest
# truncation usefully.
QUOTED_START = re.compile(r'^\s*[-•]?\s*[_#>*“"]')

# Quoted prayer lists ("(a) Admit the petition...", "(ii) Allow the
# petitioner...") are the petitioner's ask, not the Commission's holding.
PRAYER_LINE = re.compile(
    r'^\s*[-•]?\s*\(?([a-z]|i{1,3}v?|vi{0,3}|x?i{0,3})[\.\)]\s*'
    r'(admit|allow|grant|condone|permit|direct|approve|declare|pass such)',
    re.IGNORECASE,
)


def _is_table(text: str) -> bool:
    return text.count("|") >= 4


def _is_quoted(text: str) -> bool:
    return bool(QUOTED_START.match(text))


def split_paragraphs(fulltext: str) -> list[tuple[int | None, str]]:
    """Split on blank lines; drop tiny fragments (page numbers, headers)."""
    raw = re.split(r"\n\s*\n", fulltext)
    paras: list[tuple[int | None, str]] = []
    for block in raw:
        # strip pymupdf4llm inline markup (<mark>, <sup>, <br>, ...)
        block = re.sub(r"</?[a-zA-Z][^>]*>", " ", block)
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) < 40:
            continue
        m = PARA_NUM.match(block)
        paras.append((int(m.group(1)) if m else None, block))
    return paras


def strip_head(paras: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Drop cause title / coram / appearances before the substantive body.

    The respondent list in the cause title is itself numbered 1, 2, 3...,
    so 'first paragraph numbered 1' only marks the body start if that
    paragraph does NOT look like an address entry."""
    for i, (num, text) in enumerate(paras):
        if _is_address(text):
            continue
        if BODY_START.match(text) or num == 1:
            return paras[i:]
    return [p for p in paras if not HEAD_NOISE.match(p[1])]


def drop_noise(paras: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Remove address entries, table fragments, and PDF-extraction artifacts
    (pymupdf4llm picture-text blocks / page headers) that survived head-strip."""
    return [(n, t) for n, t in paras
            if not _is_address(t) and not _is_table(t)
            and "<!--" not in t
            and not re.search(r"\bPage \d+\s*(of \d+)?\s*$", t)]


# ---------------------------------------------------------------------------
# 2. Ranking: centrality + operative bonus + tail bias (+ optional query bias)
# ---------------------------------------------------------------------------

# Long tariff orders can run to hundreds of paragraphs; embedding all of them
# just to keep 8 dominates runtime. Pre-shortlist to the paragraphs that could
# plausibly win anyway: the opening, operative-language hits, and the tail.
MAX_RANK_PARAS = 80


def _shortlist(paras: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    if len(paras) <= MAX_RANK_PARAS:
        return paras
    idx: set[int] = set(range(5))                              # opening
    idx |= set(range(max(0, len(paras) - 30), len(paras)))     # tail
    for i, (_, text) in enumerate(paras):
        if OPERATIVE.search(text):
            idx.add(i)
    keep = sorted(idx)
    if len(keep) > MAX_RANK_PARAS:                             # trim the middle
        half = MAX_RANK_PARAS // 2
        keep = keep[:half] + keep[-half:]
    return [paras[i] for i in keep]

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def rank_paragraphs(
    paras: list[tuple[int | None, str]],
    model,
    query: str | None = None,
    top_k: int = 8,
) -> list[tuple[int | None, str]]:
    paras = _shortlist(paras)
    texts = [t for _, t in paras]
    emb = np.asarray(model.encode(texts, batch_size=32, show_progress_bar=False))

    centroid = emb.mean(axis=0, keepdims=True)
    scores = _cosine(emb, centroid).ravel()

    if query:
        q = np.asarray(model.encode([query]))
        scores = 0.5 * scores + 0.5 * _cosine(emb, q).ravel()

    n = len(paras)
    for i, (_, text) in enumerate(paras):
        if OPERATIVE.search(text):
            scores[i] += 0.15
        if i >= n - max(3, n // 10):
            scores[i] += 0.10   # directions / disposal live at the tail
        if i == 0:
            scores[i] += 0.05   # opening para usually states the issue
        if _is_quoted(text):
            scores[i] -= 0.25   # quoted regulations/extracts are context,
                                # not the Commission's own finding
        if PRAYER_LINE.match(text):
            scores[i] -= 0.25   # petitioner's prayers, not the holding

    keep = sorted(np.argsort(scores)[::-1][:top_k])
    return [paras[i] for i in keep]


# ---------------------------------------------------------------------------
# 3. Digest assembly
# ---------------------------------------------------------------------------

def _norm_prefix(text: str, n: int = 120) -> str:
    # digit-insensitive: corrigendum lines ("table in paragraph 78/79/105 is
    # substituted...") differ only in their numbers and should collapse
    return re.sub(r"[^a-z]", "", text.lower())[:n]


def build_digest(fulltext: str, model, query: str | None = None,
                 max_chars: int = 3500) -> str:
    paras = drop_noise(strip_head(split_paragraphs(fulltext)))
    if not paras:
        return fulltext[:max_chars]

    picked = rank_paragraphs(paras, model, query=query)

    out, used = [], 0
    seen_prefixes: set[str] = set()
    for num, text in picked:
        # Tagged petitions (e.g. "722/MP/2020 & 723/MP/2020") repeat their
        # caption verbatim; keep only the first of any near-identical pair.
        prefix = _norm_prefix(text)
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        label = f"[para {num}] " if num is not None else "[¶] "
        entry = label + text
        if used + len(entry) > max_chars:
            entry = entry[: max_chars - used].rsplit(" ", 1)[0] + " …"
        out.append(entry)
        used += len(entry)
        if used >= max_chars:
            break
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# 4. Cache-through accessor — called by the summarize_order MCP tool
# ---------------------------------------------------------------------------

def get_or_create_summary(
    conn,
    table: str,
    identifier: str,          # id hash OR petition_no (matches get_order)
    model_loader,             # the server's _get_model
    query: str | None = None,
    force_refresh: bool = False,
) -> dict:
    """Return {"id", "petition_no", "summary", "cached"}.

    Generic digests are cached in the summary column; query-biased digests
    are computed fresh (they're question-specific) but MiniLM is local so
    that's cheap. New rows loaded by the GitHub backfill are summarized
    lazily on first access — no loader change required.
    """
    if table not in TABLES:
        raise ValueError(f"table must be one of {sorted(TABLES)}")
    text_col = TABLES[table]["text_col"]

    with conn.cursor() as cur:
        cur.execute(
            f"select id, petition_no, summary, summary_model, {text_col} "
            f"from {table} where id = %s or petition_no = %s limit 1",
            (identifier, identifier),
        )
        row = cur.fetchone()
        if not row:
            return {"id": None, "summary": None, "error": "order not found"}

        oid, petition_no, cached_summary, cached_ver, fulltext = row

        if (cached_summary and cached_ver == SUMMARY_VERSION
                and not force_refresh and query is None):
            return {"id": oid, "petition_no": petition_no,
                    "summary": cached_summary, "cached": True}

        if not fulltext:
            return {"id": oid, "petition_no": petition_no, "summary": None,
                    "error": f"{text_col} is empty for this order — "
                             f"use get_order / the PDF instead"}

        digest = build_digest(fulltext, model_loader(), query=query)

        if query is None:
            cur.execute(
                f"update {table} set summary = %s, summary_model = %s, "
                f"summarized_at = now() where id = %s",
                (digest, SUMMARY_VERSION, oid),
            )
            conn.commit()

    return {"id": oid, "petition_no": petition_no,
            "summary": digest, "cached": False}
