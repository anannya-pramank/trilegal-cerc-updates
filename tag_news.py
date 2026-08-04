#!/usr/bin/env python3
"""
tag_news.py — Tier B backbone: tag news_items against the SAME subject-matter
taxonomy as orders, reusing tag_documents.py wholesale.

Why a wrapper and not a fork
----------------------------
tag_documents.py is fully config-driven by its SOURCES dict and shortlists on a
STORED pgvector embedding column. news_watch.py already gives news_items a
MiniLM `embedding` in the same space. So tagging news is just: register a 'news'
source pointing at news_items, restrict to the subject facet (news has no
petition/party/disposition), and call the tagger's own main(). Zero classifier
code duplicated — when you improve the tagger, news tagging improves too.

The shared doc_tags row (source='news', facet='subject', code=…) is exactly what
the exporter and the Tier-B issue-join read: a news item and an order relate when
they carry the same subject code. No separate news-tag table.

Usage
-----
  python tag_news.py                         # tag untagged news, subject facet
  python tag_news.py --limit 50 --skip-tagged
  python tag_news.py --backend embed         # cosine-only, fastest
  DATABASE_URL=... python tag_news.py

Any flag tag_documents.py accepts is passed straight through, except --source
(forced to 'news') and the facet scope (forced to subject) unless you override.
"""
import sys
import runpy
import importlib


def main():
    td = importlib.import_module("tag_documents")

    # Register news as a first-class source. news_items carries the same columns
    # the tagger reads (id, title, a digest-like summary, a date, an embedding).
    td.SOURCES["news"] = {
        "table": "news_items", "id": "id", "petition": "id",  # no petition; alias id
        "title": "title", "digest": "summary", "date": "published",
        "has_petition_type": False,
    }

    # News only gets subject-matter tags. Instrument/party/disposition are order
    # concepts and would be noise on a news headline. Restrict the facet set the
    # tagger iterates. (EMBED_FACETS drives the shortlist-eligible facets.)
    td.EMBED_FACETS = ["subject"]
    if hasattr(td, "FACET_ORDER"):
        td.FACET_ORDER = ["subject"]

    # Force --source news; keep every other CLI flag the user passed.
    argv = [a for a in sys.argv[1:]]
    if "--source" in argv:
        i = argv.index("--source")
        del argv[i:i + 2]
    sys.argv = ["tag_documents.py", "--source", "news"] + argv

    td.main()


if __name__ == "__main__":
    main()
