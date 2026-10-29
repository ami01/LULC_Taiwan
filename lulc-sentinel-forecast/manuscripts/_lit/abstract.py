"""Fetch abstracts for DOIs (Crossref, then Semantic Scholar, then OpenAlex).

Used to write ledger evidence from what each source actually says.
Caches to abstracts.json next to this file.

    python manuscripts/_lit/abstract.py 10.xxxx/yyyy [...]
"""
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).with_name("abstracts.json")
MAIL = "mailto=aman.mceccr@gmail.com"


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lulc-lit/0.1 (" + MAIL + ")"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def clean(t):
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def fetch(doi):
    q = urllib.parse.quote(doi)
    cr = get(f"https://api.crossref.org/works/{q}?{MAIL}")
    meta = {}
    if cr:
        m = cr["message"]
        meta = dict(title=clean((m.get("title") or [""])[0]),
                    year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    venue=(m.get("container-title") or [""])[0],
                    authors="; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", [])[:4]))
        if m.get("abstract"):
            return dict(meta, abstract=clean(m["abstract"]), src="crossref")
    s2 = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{q}?fields=title,abstract,tldr,year,venue")
    if s2 and s2.get("abstract"):
        return dict(meta, abstract=clean(s2["abstract"]), src="semanticscholar")
    oa = get(f"https://api.openalex.org/works/doi:{q}?{MAIL}")
    if oa and oa.get("abstract_inverted_index"):
        ii = oa["abstract_inverted_index"]
        w = sorted((p, k) for k, v in ii.items() for p in v)
        return dict(meta, abstract=" ".join(k for _, k in w), src="openalex")
    if s2 and s2.get("tldr"):
        return dict(meta, abstract="TLDR: " + s2["tldr"]["text"], src="s2-tldr")
    return dict(meta, abstract=None, src=None)


if __name__ == "__main__":
    c = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for d in sys.argv[1:]:
        d = d.lower()
        if d not in c or not c[d].get("abstract"):
            c[d] = fetch(d)
            CACHE.write_text(json.dumps(c, indent=1, ensure_ascii=False))
        r = c[d]
        print(f"== {d} [{r.get('src')}] {r.get('authors','')} ({r.get('year')}) {r.get('venue','')}\n   {r.get('title','')}\n   {(r.get('abstract') or 'NO ABSTRACT')[:1500]}\n")
