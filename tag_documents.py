#!/usr/bin/env python3
"""
tag_documents.py — classify CERC orders / APTEL judgments against the taxonomy.

Retrieve-then-classify, with a pluggable second stage. No API key required in
the default configuration.

  1. PETITION TYPE  — deterministic regex on petition_no (235/MP/2021 -> MP).
     Confidence 1.0, method 'regex'. CERC only.

  2. SUBJECT / INSTRUMENT / PARTY — two stages:
       a. SHORTLIST: MiniLM cosine top-k of the document's STORED embedding
          against taxonomy_nodes (pgvector). Reuses the vectors already on
          cerc_orders / aptel_orders — nothing is re-embedded. For small facets
          (<= --score-all-max nodes) we skip the shortlist and score every node,
          so recall isn't capped by the bi-encoder.
       b. DISAMBIGUATE: the chosen backend scores each candidate label.

  3. DISPOSITION — backend scores outcome (and, for APTEL, origin) labels.

BACKENDS (--backend):
  * local  (default) — MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c, an NLI
                       zero-shot classifier. Runs on CPU, NO API KEY. The "-c"
                       build is trained only on commercially-friendly data.
  * llm              — Gemini or Claude, autodetected from env keys. Needs a key.
  * embed            — pure cosine + threshold. Fastest, no second model, but
                       noisy on fine distinctions. No key.

Writes to doc_tags. Idempotent per (source, doc_id, facet). Resumable with
--skip-tagged. On CPU, 'local' is a few seconds per doc; use --deadline-minutes
so a GitHub Actions job exits cleanly and a re-run resumes.

Usage:
    export DATABASE_URL="postgresql://...:6543/postgres"

    # default: fully local, no key
    python tag_documents.py --source cerc --limit 20
    python tag_documents.py --source cerc
    python tag_documents.py --source aptel --skip-tagged --deadline-minutes 320

    # accuracy variant (slower on CPU): the large zero-shot model
    python tag_documents.py --source cerc --local-model MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c

    # if you'd rather use an API after all
    export GEMINI_API_KEY=...   # or ANTHROPIC_API_KEY
    python tag_documents.py --source cerc --backend llm
"""

import argparse
import json
import os
import re
import sys
import time

SOURCES = {
    "cerc": {
        "table": "cerc_orders", "id": "id", "petition": "petition_no",
        "title": "subject", "digest": "pdf_digest", "date": "date_order",
        "has_petition_type": True,
    },
    "aptel": {
        "table": "aptel_orders", "id": "id", "petition": "petition_no",
        "title": "cause_title", "digest": "pdf_digest", "date": "date_of_decision",
        "has_petition_type": False,
    },
}

EMBED_FACETS = ["subject", "instrument", "party"]

KNOWN_PTYPE = {
    "MP": "ptype-mp", "TT": "ptype-tt", "GT": "ptype-gt", "RP": "ptype-rp",
    "SM": "ptype-sm", "RC": "ptype-rc", "AT": "ptype-at", "TDL": "ptype-tdl",
    "FP": "ptype-fp", "L": "ptype-l", "IA": "ptype-ia",
}

# Natural-language hypothesis templates for the zero-shot / LLM stage.
FACET_TEMPLATE = {
    "subject":    "This electricity-regulatory matter concerns {}.",
    "instrument": "This order applies or interprets {}.",
    "party":      "One of the parties to this matter is {}.",
}

# Disposition is verbalised directly (labels map to codes via these dicts).
DISP_OUTCOME = {
    "allowed in full": "disp-out-1",
    "allowed in part": "disp-out-2",
    "dismissed": "disp-out-3",
    "set aside": "disp-out-4",
    "remanded to the Commission": "disp-out-5",
    "disposed of with directions": "disp-out-6",
    "an interim or interlocutory order": "disp-out-7",
}
DISP_OUTCOME_TEMPLATE = "The outcome of this matter is that it was {}."
DISP_ORIGIN = {
    "the Central Electricity Regulatory Commission": "disp-org-1",
    "a State Electricity Regulatory Commission": "disp-org-2",
    "another authority or adjudicating officer": "disp-org-3",
}
DISP_ORIGIN_TEMPLATE = "This appeal challenges an order passed by {}."


# ------------------------------------------------------------------ DB

def db_connect():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL (Supabase transaction pooler; '@' -> %40).")
    return psycopg2.connect(url)


def load_facet_nodes(conn):
    """{facet: [(code, name), ...]} for every facet. Names are the verbalised
    labels used by the zero-shot / LLM stage; also gives per-facet node counts
    used to decide shortlist-vs-score-all."""
    out = {}
    with conn.cursor() as cur:
        cur.execute("select facet, code, name from taxonomy_nodes order by facet, code")
        for facet, code, name in cur.fetchall():
            out.setdefault(facet, []).append((code, name))
    return out


def fetch_docs(conn, cfg, limit, skip_tagged, source):
    cols = f"{cfg['id']}, {cfg['petition']}, {cfg['title']}, {cfg['digest']}"
    where = ""
    params = None
    if skip_tagged:
        where = f"where {cfg['id']} not in (select doc_id from doc_tags where source = %s)"
        params = (source,)
    q = f"select {cols} from {cfg['table']} {where} order by {cfg['date']} desc nulls last"
    if limit:
        q += f" limit {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    return [dict(zip(["id", "petition", "title", "digest"], r)) for r in rows]


def shortlist(conn, doc_id, table, id_col, facet, k):
    q = f"""
        with d as (select embedding from {table} where {id_col} = %s)
        select n.code, n.name,
               1 - (n.embedding <=> (select embedding from d)) as sim
        from taxonomy_nodes n
        where n.facet = %s and n.embedding is not null
          and (select embedding from d) is not null
        order by n.embedding <=> (select embedding from d)
        limit %s;
    """
    with conn.cursor() as cur:
        cur.execute(q, (doc_id, facet, k))
        return [(c, n, float(s)) for c, n, s in cur.fetchall()]


def write_tags(conn, source, doc_id, facet, tags):
    with conn.cursor() as cur:
        cur.execute("delete from doc_tags where source=%s and doc_id=%s and facet=%s",
                    (source, doc_id, facet))
        if tags:
            cur.executemany("""
                insert into doc_tags (source, doc_id, facet, code, confidence, method, rank)
                values (%s,%s,%s,%s,%s,%s,%s)
                on conflict (source, doc_id, facet, code) do update set
                    confidence=excluded.confidence, method=excluded.method,
                    rank=excluded.rank, tagged_at=now();
            """, [(source, doc_id, facet, c, conf, m, rk) for c, conf, m, rk in tags])


# ------------------------------------------------------------------ petition type

_PET_RE     = re.compile(r"/\s*([A-Za-z]{1,4})\s*/")
_PET_RE_ALT = re.compile(r"/\s*([A-Za-z]{1,4})[\s/]*\d{4}\b")

def petition_type_tag(petition_no):
    if not petition_no:
        return None
    m = _PET_RE.search(petition_no) or _PET_RE_ALT.search(petition_no)
    if not m:
        return None
    suffix = m.group(1).upper()
    code = KNOWN_PTYPE.get(suffix, "ptype-other")
    conf = 1.0 if code != "ptype-other" else 0.5
    return code, conf


# ------------------------------------------------------------------ backends

class LocalBackend:
    """deberta-v3 NLI zero-shot classifier. Local, CPU, no API key.

    multi_label=True gives an INDEPENDENT entailment probability per label
    (they don't sum to 1) — exactly what multi-label facets need.
    """
    def __init__(self, model_name):
        from transformers import pipeline
        device = int(os.environ.get("ZEROSHOT_DEVICE", "-1"))  # -1 = CPU
        print(f"Loading local zero-shot model '{model_name}' "
              f"(first run downloads ~400MB) …")
        self.clf = pipeline("zero-shot-classification", model=model_name, device=device)

    def score(self, text, labels, template, multi_label=True):
        """-> list of (label, score) sorted desc. `labels` are natural phrases."""
        if not labels:
            return []
        out = self.clf(text, list(labels), hypothesis_template=template,
                       multi_label=multi_label)
        return list(zip(out["labels"], out["scores"]))


class LLMBackend:
    """Gemini/Claude, autodetected. Kept for parity; needs a key."""
    def __init__(self):
        self.gem = os.environ.get("GEMINI_API_KEY")
        self.ant = os.environ.get("ANTHROPIC_API_KEY")
        if not (self.gem or self.ant):
            sys.exit("--backend llm needs GEMINI_API_KEY or ANTHROPIC_API_KEY.")

    def score(self, text, labels, template, multi_label=True):
        lab_list = "\n".join(f"  - {l}" for l in labels)
        prompt = (f"For the document below, score how strongly each candidate label applies "
                  f"(0..1 independent probabilities). Template: \"{template}\"\n\nDOCUMENT:\n"
                  f"{text[:3500]}\n\nLABELS:\n{lab_list}\n\n"
                  f"Return STRICT JSON: [[\"label\", 0.0], ...]. No prose.")
        raw = self._call(prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group(0)) if m else []
        pairs = []
        for it in data:
            if isinstance(it, (list, tuple)) and it:
                pairs.append((str(it[0]), float(it[1]) if len(it) > 1 else 0.7))
        pairs.sort(key=lambda x: -x[1])
        return pairs

    def _call(self, prompt):
        try:
            if self.gem:
                import google.generativeai as genai
                genai.configure(api_key=self.gem)
                mdl = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
                return mdl.generate_content(prompt).text
            import anthropic
            client = anthropic.Anthropic(api_key=self.ant)
            msg = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=800, messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        except Exception as e:  # noqa: BLE001
            print(f"    LLM call failed ({e}); facet left to embedding fallback")
            return "[]"


# ------------------------------------------------------------------ main

def doc_text(doc):
    """Compact premise for the classifier: title carries the gist, digest the
    detail. Kept short — deberta caps at 512 tokens and shorter = better."""
    title = (doc.get("title") or "").strip()
    digest = (doc.get("digest") or "").strip()
    return (title + ". " + digest)[:1800]


def clean_label(name):
    return (name or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cerc", "aptel"], required=True)
    ap.add_argument("--backend", choices=["local", "llm", "embed"], default="local")
    ap.add_argument("--local-model", default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c",
                    help="HF zero-shot model for --backend local. Use the -large- "
                         "variant for higher accuracy at ~3x CPU cost.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-tagged", action="store_true")
    ap.add_argument("--shortlist-k", type=int, default=12,
                    help="Candidates from cosine for facets bigger than --score-all-max.")
    ap.add_argument("--score-all-max", type=int, default=35,
                    help="Facets with <= this many nodes skip the shortlist and score all "
                         "(no recall loss). Only 'subject' (64 nodes) is shortlisted by default.")
    ap.add_argument("--keep-threshold", type=float, default=0.55,
                    help="Multi-label facets: keep labels scored >= this (local/llm).")
    ap.add_argument("--embed-threshold", type=float, default=0.30,
                    help="--backend embed: keep cosine candidates with sim >= this.")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--deadline-minutes", type=int, default=0,
                    help="Soft wall-clock cap; flush and exit 0 so Actions' 6h kill "
                         "never lands mid-batch. Re-run resumes via --skip-tagged.")
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    conn = db_connect()
    nodes = load_facet_nodes(conn)

    backend = None
    if args.backend == "local":
        backend = LocalBackend(args.local_model)
    elif args.backend == "llm":
        backend = LLMBackend()

    docs = fetch_docs(conn, cfg, args.limit, args.skip_tagged, args.source)
    print(f"{len(docs)} {args.source} documents to tag via '{args.backend}'.")

    def candidates(doc_id, facet):
        """(labels, code_by_label, sim_by_label). Score-all for small facets,
        cosine shortlist for big ones."""
        all_nodes = nodes.get(facet, [])
        if len(all_nodes) <= args.score_all_max:
            return ([clean_label(n) for _, n in all_nodes],
                    {clean_label(n): c for c, n in all_nodes}, {})
        sl = shortlist(conn, doc_id, cfg["table"], cfg["id"], facet, args.shortlist_k)
        return ([clean_label(n) for _, n, _ in sl],
                {clean_label(n): c for _, n, _ in sl},
                {clean_label(n): s for _, n, s in sl})

    started = time.monotonic()
    tagged = 0
    for i, doc in enumerate(docs, 1):
        if args.deadline_minutes and (time.monotonic() - started) > args.deadline_minutes * 60:
            conn.commit()
            print(f"\n[deadline] reached after {i-1}/{len(docs)} — committed and exiting. "
                  f"Re-run with --skip-tagged to resume.")
            break

        did = doc["id"]
        print(f"[{i}/{len(docs)}] {str(doc.get('petition'))[:60]}")
        text = doc_text(doc)

        # 1. petition type
        if cfg["has_petition_type"]:
            pt = petition_type_tag(doc.get("petition"))
            if pt:
                write_tags(conn, args.source, did, "ptype", [(pt[0], pt[1], "regex", 1)])

        # 2. subject / instrument / party
        for facet in EMBED_FACETS:
            labels, code_by_label, sim_by_label = candidates(did, facet)
            if not labels:
                continue
            if args.backend == "embed":
                keep = sorted(((l, sim_by_label.get(l, 0.0)) for l in labels),
                              key=lambda x: -x[1])
                tags = [(code_by_label[l], round(s, 3), "embed", r + 1)
                        for r, (l, s) in enumerate(keep) if s >= args.embed_threshold]
            else:
                scored = backend.score(text, labels, FACET_TEMPLATE[facet], multi_label=True)
                method = "embed+local" if args.backend == "local" else "embed+llm"
                tags = [(code_by_label[l], round(s, 3), method, r + 1)
                        for r, (l, s) in enumerate(scored)
                        if l in code_by_label and s >= args.keep_threshold]
                if not tags and scored:  # never leave a facet empty: keep top-1
                    l, s = scored[0]
                    if l in code_by_label:
                        tags = [(code_by_label[l], round(s, 3), method, 1)]
            write_tags(conn, args.source, did, facet, tags)

        # 3. disposition (skip under embed-only: outcomes aren't separable in cosine space)
        if args.backend != "embed":
            disp_tags = []
            out = backend.score(text, list(DISP_OUTCOME.keys()),
                                DISP_OUTCOME_TEMPLATE, multi_label=False)
            if out:
                l, s = out[0]
                disp_tags.append((DISP_OUTCOME[l], round(s, 3), args.backend, 1))
            if args.source == "aptel":
                org = backend.score(text, list(DISP_ORIGIN.keys()),
                                    DISP_ORIGIN_TEMPLATE, multi_label=False)
                if org:
                    l, s = org[0]
                    disp_tags.append((DISP_ORIGIN[l], round(s, 3), args.backend, 2))
            write_tags(conn, args.source, did, "disposition", disp_tags)

        tagged += 1
        if i % args.batch == 0:
            conn.commit()
            print(f"    … committed ({tagged} tagged)")

    conn.commit()
    conn.close()
    print(f"Done. Tagged {tagged} {args.source} documents via '{args.backend}'.")


if __name__ == "__main__":
    main()
