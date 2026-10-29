"""Housekeeping for crossref-bib output. Never edits metadata.

    python _lit/bibtool.py keys  refs.bib          # rename unusable keys
    python _lit/bibtool.py prune refs.bib paper.tex # keep only cited entries

`keys` renames entries whose key is a URL or a bare year (DataCite records),
using a fixed map by DOI, so \\cite keys are stable and LaTeX-safe.
"""
import re
import sys

KEYMAP = {
    "10.5281/zenodo.7254221": "Zanaga_2022",
    "10.5270/esa-c5d3d65": "CopernicusDEM_2022",
    "10.24381/cds.006f2c9a": "C3S_LandCover_2019",
    "10.2909/960998c1-1870-4e82-8051-6485205ebbac": "EEA_CLC2018_2019",
}


def entries(src):
    starts = [m.start() for m in re.finditer(r"(?m)^@\w+\s*\{", src)]
    return [src[a:b] for a, b in zip(starts, starts[1:] + [len(src)])]


def key_of(e):
    return re.match(r"@\w+\s*\{\s*([^,\s]+)", e).group(1)


def doi_of(e):
    m = re.search(r"\bdoi\s*=\s*[{\"]\s*([^}\"]+)", e, re.I)
    return m.group(1).strip().lower() if m else None


def cmd_keys(bib):
    src = open(bib, encoding="utf-8").read()
    out = []
    for e in entries(src):
        k, d = key_of(e), doi_of(e)
        if d in KEYMAP and k != KEYMAP[d]:
            e = e.replace(k, KEYMAP[d], 1)
        out.append(e)
    open(bib, "w", encoding="utf-8").write("".join(out))


def cmd_prune(bib, tex):
    t = re.sub(r"(?<!\\)%.*", "", open(tex, encoding="utf-8").read())
    cited = set()
    for m in re.finditer(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]+)\}", t):
        cited |= {k.strip() for k in m.group(1).split(",")}
    src = open(bib, encoding="utf-8").read()
    es = entries(src)
    keep = [e for e in es if key_of(e) in cited]
    open(bib, "w", encoding="utf-8").write("".join(keep))
    missing = cited - {key_of(e) for e in es}
    print(f"kept {len(keep)} of {len(es)}; cited but missing: {sorted(missing)}")


if __name__ == "__main__":
    {"keys": cmd_keys, "prune": cmd_prune}[sys.argv[1]](*sys.argv[2:])
