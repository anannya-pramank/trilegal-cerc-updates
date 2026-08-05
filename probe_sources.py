#!/usr/bin/env python3
"""
probe_sources.py — figure out what each site actually allows for backfill.

The first backfill run showed the three sites behave differently (404 / 403 /
non-JSON). Before committing to an acquisition method, this probes several per
site and prints a verdict. Run it in Actions (or anywhere with outbound access)
and paste the output back.

For each site it tries, in order:
  1. WP REST posts           /wp-json/wp/v2/posts?per_page=1
  2. WP REST root            /wp-json/                     (is REST enabled at all?)
  3. WP feed (paged)         /feed/?paged=2                (RSS with pagination?)
  4. sitemap index           /sitemap.xml or /sitemap_index.xml
  5. news sitemap            common WP/Yoast sitemap paths
  6. plain homepage GET      (does anything respond 200 without a block?)

It reports status code, content-type, and a short body sniff for each, plus a
one-line verdict per site. No writes, no deps beyond requests.

Usage:
  python probe_sources.py
  python probe_sources.py --ua browser   # try a browser-like User-Agent
"""
import argparse
import sys

import requests

SITES = {
    "saur":       "https://www.saurenergy.com",
    "energetica": "https://www.energetica-india.net",
    "mercom":     "https://www.mercomindia.com",
}

UA_BOT = "cerc-atlas-backfill/1.0 (+regulatory research)"
UA_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def probe(url, headers, label):
    try:
        r = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
    except Exception as e:
        print(f"    {label:16} ERROR {type(e).__name__}: {e}")
        return None
    ctype = r.headers.get("Content-Type", "")
    body = (r.text or "")[:120].replace("\n", " ")
    is_json = "json" in ctype.lower()
    is_xml = "xml" in ctype.lower() or body.lstrip().startswith("<?xml")
    print(f"    {label:16} {r.status_code}  {ctype[:40]:40}  "
          f"{'JSON' if is_json else 'XML' if is_xml else 'other':5}  {body[:60]!r}")
    return r


def verdict(results):
    """results: dict label->Response|None. Return a one-line recommendation."""
    def ok(label):
        r = results.get(label)
        return r is not None and r.status_code == 200
    def json_ok(label):
        r = results.get(label)
        return r is not None and r.status_code == 200 and "json" in \
            r.headers.get("Content-Type", "").lower()
    if json_ok("wp_posts"):
        return "USE wp-rest (news_backfill.py works as-is)"
    if ok("wp_root"):
        return "WP REST enabled but posts blocked — try again w/ browser UA, else feed"
    if ok("feed_paged"):
        return "USE paged RSS feed backfill (feed/?paged=N) — I'll write that path"
    if ok("sitemap_news") or ok("sitemap_index") or ok("sitemap"):
        return "USE sitemap walk — I'll write that path"
    if any(results.get(k) is not None and results[k].status_code == 403
           for k in results):
        return "BLOCKED (403) — bot shield; needs browser UA or a different source"
    return "no obvious archive path — inspect manually"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ua", choices=["bot", "browser"], default="bot")
    args = ap.parse_args()
    ua = UA_BROWSER if args.ua == "browser" else UA_BOT
    headers = {"User-Agent": ua, "Accept": "*/*"}
    print(f"probing with UA={args.ua}\n")

    for sid, base in SITES.items():
        print(f"[{sid}] {base}")
        results = {}
        results["wp_posts"]      = probe(f"{base}/wp-json/wp/v2/posts?per_page=1", headers, "wp_posts")
        results["wp_root"]       = probe(f"{base}/wp-json/", headers, "wp_root")
        results["feed_paged"]    = probe(f"{base}/feed/?paged=2", headers, "feed_paged")
        results["sitemap"]       = probe(f"{base}/sitemap.xml", headers, "sitemap")
        results["sitemap_index"] = probe(f"{base}/sitemap_index.xml", headers, "sitemap_index")
        results["sitemap_news"]  = probe(f"{base}/news-sitemap.xml", headers, "sitemap_news")
        results["homepage"]      = probe(f"{base}/", headers, "homepage")
        print(f"  VERDICT: {verdict(results)}\n")


if __name__ == "__main__":
    main()
