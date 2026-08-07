#!/usr/bin/env python3
"""
entity_resolver.py — one source of truth for entity identity.

Loads entities.yaml and resolves any surface form (an order's party string, a
news item's text, or a search query) to a canonical entity id. Distinct entities
stay distinct: "Adani Power Ltd" -> adani_power, "Adani Green" -> adani_green,
bare "Adani" -> adani_generic.

Matching rules
--------------
* Normalize: lowercase, strip corporate suffixes/noise, collapse whitespace.
* Try aliases LONGEST-first across all entities, so a specific alias
  ("adani green energy") wins over a generic one ("adani").
* Honor `exclude`: an entity won't match if any of its exclude terms is present
  (this is how bare "adani" avoids stealing "adani power").
* An input can resolve to MULTIPLE entities (a cause title naming two parties) —
  resolve_all returns every distinct hit.
"""
import re
from pathlib import Path

import yaml

_SUFFIX = re.compile(
    r"\b(ltd|limited|pvt|private|corp|corporation|co|company|inc|"
    r"m/s|the)\b", re.I,
)
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(s):
    if not s:
        return ""
    s = _NONWORD.sub(" ", s.lower())
    s = _SUFFIX.sub(" ", s)
    return _WS.sub(" ", s).strip()


class EntityRegistry:
    def __init__(self, path="entities.yaml"):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.entities = {}          # id -> {name, type, aliases[], exclude[]}
        self._alias_index = []      # (normalized_alias, entity_id, has_exclude)
        for e in cfg["entities"]:
            eid = e["id"]
            self.entities[eid] = {
                "id": eid, "name": e["name"], "type": e.get("type", "other"),
                "aliases": e.get("aliases", []),
                "exclude": [normalize(x) for x in e.get("exclude", [])],
                "patterns": [re.compile(p, re.I) for p in e.get("patterns", [])],
            }
            for a in e.get("aliases", []):
                na = normalize(a)
                if na:
                    self._alias_index.append((na, eid))
        # longest alias first: specific beats generic
        self._alias_index.sort(key=lambda x: -len(x[0]))

    def _excluded(self, eid, ntext):
        return any(x in ntext for x in self.entities[eid]["exclude"])

    def resolve_all(self, text):
        """Return ordered list of distinct entity ids present in text."""
        if not text:
            return []
        nt = " " + normalize(text) + " "
        found = []
        claimed_spans = []  # avoid a generic alias re-matching inside a specific one
        for alias, eid in self._alias_index:
            pad = " " + alias + " "
            idx = nt.find(pad)
            if idx == -1:
                # also allow alias at string start/end boundaries
                if nt.strip() == alias or nt.startswith(alias + " ") or nt.endswith(" " + alias):
                    idx = nt.find(alias)
                else:
                    continue
            if self._excluded(eid, nt):
                continue
            if eid not in found:
                # check this hit isn't wholly inside an already-claimed longer alias
                span = (idx, idx + len(alias))
                inside = any(s0 <= span[0] and span[1] <= s1 for s0, s1 in claimed_spans)
                if inside:
                    continue
                found.append(eid)
                claimed_spans.append(span)
        # regex patterns (rare)
        for eid, e in self.entities.items():
            if eid in found:
                continue
            if any(p.search(text) for p in e["patterns"]) and not self._excluded(eid, nt):
                found.append(eid)
        return found

    def resolve_one(self, text):
        r = self.resolve_all(text)
        return r[0] if r else None

    def resolve_query(self, q):
        """For the search bar: which entity does a typed query mean?
        Returns entity id or None. Prefers an exact-ish alias/name match."""
        nq = normalize(q)
        if not nq:
            return None
        # exact canonical-name match
        for eid, e in self.entities.items():
            if normalize(e["name"]) == nq:
                return eid
        # exact alias match
        for alias, eid in self._alias_index:
            if alias == nq and not self._excluded(eid, " " + nq + " "):
                return eid
        # fall back to substring resolution
        return self.resolve_one(q)


if __name__ == "__main__":
    # quick self-test
    import sys
    reg = EntityRegistry(sys.argv[1] if len(sys.argv) > 1 else "entities.yaml")
    for t in ["Adani Power Ltd", "Adani Power Limited", "Adani Green Energy Ltd",
              "M/s Adani Transmission", "Adani", "NTPC Ltd",
              "Adani Power Rajasthan Ltd v. NTPC Ltd"]:
        print(f"{t!r:45} -> {reg.resolve_all(t)}")
