"""
CERC updates scraper — orders + regulations.

Connectivity notes (see commit history / earlier debugging):
  * Use the www host. https://www.cercind.gov.in negotiates modern TLS, so plain
    `requests` talks to it directly. The bare apex (cercind.gov.in) only offers
    legacy ciphers that modern OpenSSL rejects (SECLEVEL=2) — that, NOT any
    geofence, was the original failure. The site is reachable from GitHub's
    US-hosted runners, so no proxy / Indian egress is needed.
  * As a safety net, fetch() falls back to curl with TLS 1.2 + SECLEVEL=1 if it
    ever hits an SSL error (e.g. someone points it at the apex again).

Dependencies:
  Lean (default):   requests  beautifulsoup4  pdfplumber
  Optional upgrade: set CERC_SEMANTIC=1 and also install  sentence-transformers  numpy
                    to use embedding-based extraction on very long PDFs. Without
                    it, a dependency-free keyword heuristic is used instead.
"""

import os
import re
import io
import csv
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# ================= CONFIG =================

DATA_DIR        = Path("cerc")
ORDERS_CSV      = DATA_DIR / "cerc_orders_master.csv"
ORDERS_JSON     = DATA_DIR / "cerc_orders_new.json"
REGS_CSV        = DATA_DIR / "cerc_regs_master.csv"
REGS_JSON       = DATA_DIR / "cerc_regs_new.json"

# www host: modern TLS, works with plain requests, no proxy required.
CERC_BASE       = "https://www.cercind.gov.in"
ORDERS_URL      = f"{CERC_BASE}/recent_orders.html"            # current year (the live page)
ORDERS_URL_TMPL = f"{CERC_BASE}/recent_orders{{year}}.html"    # archives, e.g. ...orders2025.html
REGS_URL        = f"{CERC_BASE}/current_reg.html"

HARD_CAP        = 80_000
MAX_PDF_PAGES   = 120          # parse deep enough to see the operative order in long tariff orders
TIMEOUT         = 40

# Size of the relevance-ranked digest emitted per item (chars). This is what the
# Power Automate feed carries — kept small so the JSON stays light and the
# notification is skimmable. Bump it if you want more detail per update.
DIGEST_CHARS    = int(os.environ.get("CERC_DIGEST_CHARS", "6000"))

# Optionally archive the FULL extracted text to cerc/fulltext/<id>.txt (kept OUT
# of the feed JSON so Power Automate isn't bloated). Off by default.
KEEP_FULLTEXT   = os.environ.get("CERC_KEEP_FULLTEXT") == "1"

# How many of the newest *new* items to actually fetch+extract per run.
# 0 = no limit (process every new item). 1 = latest only, 5 = latest five, etc.
MAX_NEW_ORDERS  = int(os.environ.get("CERC_MAX_ORDERS", "0"))
MAX_NEW_REGS    = int(os.environ.get("CERC_MAX_REGS", "0"))

# Seed mode: record every currently-listed ID into the ledger WITHOUT downloading
# any PDFs, then exit each scraper. Run once (CERC_SEED=1) to skip the first-run
# backlog so later runs only pick up genuinely new items.
SEED_ONLY       = os.environ.get("CERC_SEED") == "1"

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

SUMMARY_QUERIES = [
    "What is the final order, decision, or direction issued?",
    "What are the main legal issues and questions decided in this case?",
    "What is the reasoning and legal basis for the decision?",
    "Who are the parties and what is the nature of the dispute or petition?",
]

# Operative / salient legal language, used by the no-ML extraction fallback.
LEGAL_KEYWORDS = (
    "order", "ordered", "directed", "direction", "hereby", "held", "decided",
    "disposed", "allowed", "dismissed", "granted", "rejected", "in view of",
    "petitioner", "respondent", "applicant", "commission", "tariff",
    "regulation", "section", "clause", "conclusion", "accordingly",
)


# ================= HTTP =================

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT,
                      "Accept-Language": "en-US,en;q=0.9"})
    if Retry is not None:
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]))
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    return s


SESSION = _make_session()


def _curl_fetch(url: str, binary: bool, timeout: int):
    """Legacy-TLS fallback. cercind apex only negotiates old ciphers."""
    cmd = [
        "curl", "-sS", "-L", "-k",
        "--tls-max", "1.2", "--ciphers", "DEFAULT@SECLEVEL=1",
        "-A", USER_AGENT, "--max-time", str(timeout), url,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed ({r.returncode}): "
                           f"{r.stderr.decode(errors='replace')}")
    return r.stdout if binary else r.stdout.decode("utf-8", errors="replace")


def fetch(url: str, binary: bool = False, timeout: int = TIMEOUT):
    """GET via requests; fall back to curl+legacy-TLS only on SSL errors."""
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content if binary else resp.text
    except requests.exceptions.SSLError:
        print(f"  [TLS fallback via curl] {url}")
        return _curl_fetch(url, binary, timeout)


# ================= PDF TEXT EXTRACTION =================

def _chunks(text: str) -> list:
    return [c.strip() for c in re.split(r"\n{2,}", text) if len(c.strip()) > 80]


def _take_to_budget(chunks: list, ranked_idx: list, budget: int) -> str:
    """Given chunk indices sorted best-first, keep them until the char budget is
    hit, then re-order to document position so the digest reads naturally."""
    selected, used = [], 0
    for i in ranked_idx:
        if used + len(chunks[i]) > budget and selected:
            break
        selected.append(i)
        used += len(chunks[i])
    return "\n\n".join(chunks[i] for i in sorted(selected))


def _keyword_rank(chunks: list) -> list:
    """No-ML fallback ranking: by density of operative legal language."""
    scores = [sum(c.lower().count(k) for k in LEGAL_KEYWORDS) for c in chunks]
    return sorted(range(len(chunks)), key=lambda i: (-scores[i], i))


def _semantic_rank(chunks: list) -> list:
    """Embedding ranking (the watcher's method): score each chunk by max cosine
    similarity to the legal SUMMARY_QUERIES. Runs fully locally — nothing leaves
    the machine; only a one-time (cached) model download touches the network."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not hasattr(_semantic_rank, "_model"):
        print("  Loading semantic model …")
        _semantic_rank._model = SentenceTransformer("all-MiniLM-L6-v2")
    model = _semantic_rank._model

    chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)
    query_embs = model.encode(SUMMARY_QUERIES, show_progress_bar=False)
    chunk_unit = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)

    scores = np.zeros(len(chunks))
    for q_emb in query_embs:
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        scores = np.maximum(scores, chunk_unit @ q_unit)
    return [int(i) for i in np.argsort(scores)[::-1]]


def make_digest(full_text: str) -> str:
    """Compact, relevance-ranked extract for the notification feed.
    Embedding ranking by default; keyword ranking if the model can't load."""
    chunks = _chunks(full_text)
    if not chunks:
        return full_text[:DIGEST_CHARS]
    if sum(len(c) for c in chunks) <= DIGEST_CHARS:   # short orders: keep as-is
        return full_text.strip()
    try:
        ranked = _semantic_rank(chunks)
    except Exception as e:
        print(f"  [semantic unavailable: {e}] using keyword ranking")
        ranked = _keyword_rank(chunks)
    return _take_to_budget(chunks, ranked, DIGEST_CHARS)


def _pdf_to_text(content: bytes) -> str:
    """Extract document text from raw PDF bytes.

    Primary: pymupdf4llm -> Markdown, which detects tables and reconstructs them
    as markdown grids (keeps tariff-table row labels attached to their figures)
    and preserves reading order. Fallback: pdfplumber's plain-text extraction if
    pymupdf4llm isn't available or errors on a file. Both run fully locally.
    """
    try:
        try:
            import pymupdf
        except ImportError:                       # older PyMuPDF exposes 'fitz'
            import fitz as pymupdf
        import pymupdf4llm

        doc = pymupdf.open(stream=content, filetype="pdf")
        pages = list(range(min(doc.page_count, MAX_PDF_PAGES)))
        # use_ocr=False: this pymupdf4llm version defaults to OCR-ing EVERY page,
        # which is slow and worse than the embedded text layer that CERC PDFs have.
        # Disabling it reads the real text directly. (Flip on only for scanned PDFs.)
        md = pymupdf4llm.to_markdown(doc, pages=pages, use_ocr=False)
        doc.close()
        if md and md.strip():
            return md.strip()
    except Exception as e:
        print(f"  [pymupdf4llm failed: {e}] falling back to pdfplumber")

    if HAS_PDFPLUMBER:
        parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:MAX_PDF_PAGES]:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts).strip()
    return ""


def extract_digest(pdf_url: str) -> str:
    """Download a PDF, extract text, return a relevance-ranked digest. Optionally
    archives the full text to cerc/fulltext/<id>.md (kept OUT of the feed)."""
    try:
        content = fetch(pdf_url, binary=True)
        full_text = _pdf_to_text(content)
        if not full_text:
            return ""

        if KEEP_FULLTEXT:
            ft_dir = DATA_DIR / "fulltext"
            ft_dir.mkdir(parents=True, exist_ok=True)
            (ft_dir / f"{make_id(pdf_url)}.md").write_text(full_text, encoding="utf-8")

        return make_digest(full_text)
    except Exception as e:
        print(f"  [PDF extract error] {pdf_url}: {e}")
        return ""


# ================= HELPERS =================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_date(s: str):
    """Best-effort parse of CERC date strings; returns datetime.min on failure
    so undated rows sort last."""
    s = (s or "").strip()
    for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.min


def make_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def clean(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def absolute_url(href: str) -> str:
    href = href.strip()
    return href if href.startswith("http") else CERC_BASE + "/" + href.lstrip("/")


# ================= ORDERS SCRAPER =================

def resolve_orders_url() -> str:
    # Default: the live current-year page (recent_orders.html).
    # To deliberately scrape a past year's archive, set CERC_ORDERS_YEAR=2025
    # (uses recent_orders2025.html). No silent fallback to another year — if the
    # page is wrong/missing the run fails loudly rather than scraping last year.
    year = os.environ.get("CERC_ORDERS_YEAR", "").strip()
    return ORDERS_URL_TMPL.format(year=year) if year else ORDERS_URL


def scrape_orders() -> list:
    url = resolve_orders_url()
    print(f"  Using orders page: {url}")
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    target = next((t for t in soup.find_all("table")
                   if "Petition No." in t.get_text()), None)
    if not target:
        print(f"ERROR: CERC orders table not found at {url}")
        return []

    results = []
    for tr in target.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 6:
            continue
        subject_cell = cells[2]
        pdf_tag = subject_cell.find("a", href=True)
        if not pdf_tag:
            continue
        pdf_href = pdf_tag.get("href", "").strip()
        if not pdf_href.lower().endswith(".pdf"):
            continue
        results.append({
            "petition_no": clean(cells[1]),
            "subject":     clean(subject_cell),
            "date_order":  clean(cells[3]),
            "date_posted": clean(cells[4]),
            "category":    clean(cells[5]),
            "pdf_url":     absolute_url(pdf_href),
        })
    return results


# ================= REGULATIONS SCRAPER =================

_GAZ_PATTERNS  = ["gaz", "gazette", "-gz-", "/gz-"]
_SKIP_PATTERNS = _GAZ_PATTERNS + ["sor", "statement-of", "corri", "errata",
                                  "addendum", "consolidated", "amendment_2007",
                                  "amendment_2008"]


def _pick_main_pdf(reg_cell) -> tuple:
    pdf_links = reg_cell.find_all("a", href=lambda h: h and h.lower().endswith(".pdf"))
    if not pdf_links:
        return "", ""
    gazette_url = next((absolute_url(a["href"]) for a in pdf_links
                        if any(p in a["href"].lower() for p in _GAZ_PATTERNS)), "")
    for a in pdf_links:
        if "noti" in a["href"].lower():
            return absolute_url(a["href"]), gazette_url
    for a in pdf_links:
        if not any(p in a["href"].lower() for p in _SKIP_PATTERNS):
            return absolute_url(a["href"]), gazette_url
    return absolute_url(pdf_links[0]["href"]), gazette_url


def scrape_regulations() -> list:
    html = fetch(REGS_URL)
    soup = BeautifulSoup(html, "html.parser")
    target = next((t for t in soup.find_all("table")
                   if "Gazette" in t.get_text()), None)
    if not target:
        print("ERROR: CERC regulations table not found")
        return []

    results = []
    for tr in target.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        sl_no_text = clean(cells[0]).rstrip(".")
        if not sl_no_text.isdigit():
            continue
        noti_url, gazette_url = _pick_main_pdf(cells[1])
        if not noti_url:
            continue
        reg_name_raw = cells[1].get_text(" ", strip=True)
        reg_name = re.split(r"\d\.\s+(?:Gazette|Notification|Guidelines)",
                            reg_name_raw)[0].strip()
        reg_name = re.sub(r"\s+", " ", reg_name)
        results.append({
            "sl_no":        int(sl_no_text),
            "reg_name":     reg_name,
            "gazette_no":   clean(cells[2]),
            "gazette_date": clean(cells[3]),
            "noti_pdf_url": noti_url,
            "gaz_pdf_url":  gazette_url,
        })
    return results


# ================= CSV / JSON =================

def ensure_csv(csv_path: Path, fieldnames: list):
    DATA_DIR.mkdir(exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fieldnames)


def load_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


def append_to_csv(csv_path: Path, rows: list, fieldnames: list):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames,
                       extrasaction="ignore").writerows(rows)


def write_json(json_path: Path, items: list):
    json_path.write_text(
        json.dumps({"generated_at": now_iso(), "count": len(items),
                    "items": items}, indent=2, ensure_ascii=False),
        encoding="utf-8")


# ================= MAIN =================

def _process(label, scraped, csv_path, json_path, fields, pdf_field, name_fn, limit):
    ensure_csv(csv_path, fields)
    seen = load_ids(csv_path)

    candidates = [e for e in scraped if make_id(e[pdf_field]) not in seen]
    print(f"  {len(candidates)} new (of {len(scraped)} listed)")

    # Seed mode: record IDs as seen, download nothing, then stop.
    if SEED_ONLY:
        rows = [{"id": make_id(e[pdf_field]), **e, "scraped_at": now_iso()}
                for e in candidates]
        if rows:
            append_to_csv(csv_path, rows, fields)
        print(f"  Seeded {len(rows)} {label} IDs (no PDFs fetched)")
        return

    # Newest first, then optionally keep only the latest N.
    candidates.sort(
        key=lambda e: parse_date(e.get("date_posted") or e.get("gazette_date", "")),
        reverse=True)
    if limit:
        candidates = candidates[:limit]
        print(f"  Limited to newest {len(candidates)} ({label})")

    new = []
    for e in candidates:
        print(f"  NEW {label}: {name_fn(e)}")
        new.append({"id": make_id(e[pdf_field]), **e,
                    "pdf_digest": extract_digest(e[pdf_field]),
                    "scraped_at": now_iso()})

    print(f"  Wrote {len(new)} {label}")
    if new:
        append_to_csv(csv_path, new, fields)
        write_json(json_path, new)


def run_orders():
    print("Scraping CERC orders …")
    _process(
        "order", scrape_orders(), ORDERS_CSV, ORDERS_JSON,
        ["id", "petition_no", "pdf_url", "scraped_at"],
        pdf_field="pdf_url",
        name_fn=lambda e: e["petition_no"],
        limit=MAX_NEW_ORDERS,
    )


def run_regulations():
    print("\nScraping CERC regulations …")
    _process(
        "regulation", scrape_regulations(), REGS_CSV, REGS_JSON,
        ["id", "sl_no", "reg_name", "noti_pdf_url", "scraped_at"],
        pdf_field="noti_pdf_url",
        name_fn=lambda e: f"[{e['sl_no']}] {e['reg_name'][:70]}",
        limit=MAX_NEW_REGS,
    )


def main():
    run_orders()
    run_regulations()
    print("\nAll done.")


if __name__ == "__main__":
    main()
