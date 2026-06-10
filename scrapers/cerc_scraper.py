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
ORDERS_URL_TMPL = f"{CERC_BASE}/recent_orders{{year}}.html"
REGS_URL        = f"{CERC_BASE}/current_reg.html"

HARD_CAP        = 80_000
MAX_PDF_PAGES   = 40
TIMEOUT         = 40

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


def _keyword_select(chunks: list) -> str:
    """Dependency-free fallback: keep chunks richest in operative legal language,
    always retaining the opening (parties/context) and the closing chunks, where
    the operative order usually sits."""
    scores = []
    for c in chunks:
        low = c.lower()
        scores.append(sum(low.count(k) for k in LEGAL_KEYWORDS))

    n = len(chunks)
    must_keep = {0, n - 1, max(0, n - 2)}
    # Visit must-keep first, then highest-scoring, then document order.
    order = sorted(range(n), key=lambda i: (i not in must_keep, -scores[i], i))

    selected, used = set(), 0
    for i in order:
        if i not in must_keep and used + len(chunks[i]) > HARD_CAP:
            continue
        selected.add(i)
        used += len(chunks[i])
        if used >= HARD_CAP:
            break
    return "\n\n".join(chunks[i] for i in sorted(selected))


def _semantic_select(chunks: list) -> str:
    """Embedding-based selection. Requires CERC_SEMANTIC=1 plus
    sentence-transformers + numpy (and the torch they pull in)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not hasattr(_semantic_select, "_model"):
        print("  Loading semantic model …")
        _semantic_select._model = SentenceTransformer("all-MiniLM-L6-v2")
    model = _semantic_select._model

    chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)
    query_embs = model.encode(SUMMARY_QUERIES, show_progress_bar=False)
    chunk_unit = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)

    scores = np.zeros(len(chunks))
    for q_emb in query_embs:
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        scores = np.maximum(scores, chunk_unit @ q_unit)

    threshold = 0.35 * scores.max()
    selected, used = [], 0
    for idx in range(len(chunks)):
        if scores[idx] >= threshold and used + len(chunks[idx]) <= HARD_CAP:
            selected.append(idx)
            used += len(chunks[idx])
    if not selected:  # threshold too aggressive — take top 10
        selected = sorted(int(i) for i in np.argsort(scores)[::-1][:10])

    print(f"  Semantic selection: {len(selected)}/{len(chunks)} chunks")
    return "\n\n".join(chunks[i] for i in selected)


def select_relevant(full_text: str) -> str:
    chunks = _chunks(full_text)
    if not chunks:
        return full_text[:HARD_CAP]
    if sum(len(c) for c in chunks) <= HARD_CAP:
        return full_text
    if os.environ.get("CERC_SEMANTIC") == "1":
        try:
            return _semantic_select(chunks)
        except Exception as e:
            print(f"  [semantic unavailable: {e}] using keyword selection")
    return _keyword_select(chunks)


def extract_pdf_text(pdf_url: str) -> str:
    if not HAS_PDFPLUMBER:
        return ""
    try:
        content = fetch(pdf_url, binary=True)
        parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:MAX_PDF_PAGES]:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return select_relevant("\n\n".join(parts).strip())
    except Exception as e:
        print(f"  [PDF extract error] {pdf_url}: {e}")
        return ""


# ================= HELPERS =================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def clean(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def absolute_url(href: str) -> str:
    href = href.strip()
    return href if href.startswith("http") else CERC_BASE + "/" + href.lstrip("/")


# ================= ORDERS SCRAPER =================

def resolve_orders_url() -> str:
    year = datetime.now(timezone.utc).year
    for y in (year, year - 1):
        url = ORDERS_URL_TMPL.format(year=y)
        try:
            html = fetch(url, timeout=20)
            if len(html) > 5000:
                print(f"  Using orders page: {url}")
                return url
        except Exception as e:
            print(f"  {url} -> {e}")
    raise RuntimeError("Could not resolve a valid CERC orders URL")


def scrape_orders() -> list:
    html = fetch(resolve_orders_url())
    soup = BeautifulSoup(html, "html.parser")
    target = next((t for t in soup.find_all("table")
                   if "Petition No." in t.get_text()), None)
    if not target:
        print("ERROR: CERC orders table not found")
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

def run_orders():
    fields = ["id", "petition_no", "pdf_url", "scraped_at"]
    ensure_csv(ORDERS_CSV, fields)
    seen = load_ids(ORDERS_CSV)

    print("Scraping CERC orders …")
    scraped = scrape_orders()
    print(f"  {len(scraped)} entries on page")

    new = []
    for e in scraped:
        item_id = make_id(e["pdf_url"])
        if item_id in seen:
            continue
        print(f"  NEW order: {e['petition_no']}")
        new.append({"id": item_id, **e,
                    "pdf_text": extract_pdf_text(e["pdf_url"]),
                    "scraped_at": now_iso()})

    print(f"  New orders: {len(new)}")
    if new:
        append_to_csv(ORDERS_CSV, new, fields)
        write_json(ORDERS_JSON, new)


def run_regulations():
    fields = ["id", "sl_no", "reg_name", "noti_pdf_url", "scraped_at"]
    ensure_csv(REGS_CSV, fields)
    seen = load_ids(REGS_CSV)

    print("\nScraping CERC regulations …")
    scraped = scrape_regulations()
    print(f"  {len(scraped)} entries on page")

    new = []
    for e in scraped:
        item_id = make_id(e["noti_pdf_url"])
        if item_id in seen:
            continue
        print(f"  NEW regulation: [{e['sl_no']}] {e['reg_name'][:70]}")
        new.append({"id": item_id, **e,
                    "pdf_text": extract_pdf_text(e["noti_pdf_url"]),
                    "scraped_at": now_iso()})

    print(f"  New regulations: {len(new)}")
    if new:
        append_to_csv(REGS_CSV, new, fields)
        write_json(REGS_JSON, new)


def main():
    run_orders()
    run_regulations()
    print("\nAll done.")


if __name__ == "__main__":
    main()
