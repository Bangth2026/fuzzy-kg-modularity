#!/usr/bin/env python3
"""Command-line front end of Algorithm 2 (fuzzykg.audit).

Input: an edge list CSV with columns src,dst,mu (integer node ids or labels) and optionally a
node table (id,label), a grouping JSON {label: group} and a CSV of external prerequisite pairs
(columns pre, post, score: the score that `pre` is a prerequisite of `post`).

  python scripts/fkg_audit.py data/graphs/junyi_transition_edges.csv \
      --nodes data/graphs/junyi_transition_nodes.csv --pairs pairs.csv --out audit_junyi

writes audit_junyi.json and audit_junyi.md.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph  # noqa: E402
from fuzzykg.audit import audit, report_markdown  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges"); ap.add_argument("--nodes"); ap.add_argument("--grouping"); ap.add_argument("--pairs")
    ap.add_argument("--no-null-test", action="store_true"); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--out", default="audit")
    ap.add_argument("--unit", choices=("meaningful", "arbitrary"),
                    help="unit of mu_E: meaningful (a count share, or a released or calibrated score) or arbitrary "
                         "(a weighted combination whose weights the builder chose); decides step 3 where the two drops disagree")
    a = ap.parse_args()
    e = pd.read_csv(a.edges)
    if a.nodes:
        lab = pd.read_csv(a.nodes).label.astype(str).tolist()
    else:
        lab = sorted(set(e.src.astype(str)) | set(e.dst.astype(str)))
        ix = {l: i for i, l in enumerate(lab)}; e = e.assign(src=e.src.astype(str).map(ix), dst=e.dst.astype(str).map(ix))
    G = FuzzyGraph.from_edges(zip(e.src, e.dst, e.mu), n=len(lab), labels=lab)
    grp = None
    if a.grouping:
        g = json.load(open(a.grouping)); grp = np.array([g.get(l) for l in lab], dtype=object)
    pairs = None
    if a.pairs:
        p = pd.read_csv(a.pairs); idx = {l: i for i, l in enumerate(lab)}
        p = p[p.pre.astype(str).isin(idx) & p.post.astype(str).isin(idx)]
        pairs = (p.pre.astype(str).map(idx).to_numpy(), p.post.astype(str).map(idx).to_numpy(), p.score.to_numpy(float))
    rep = audit(G, grouping=grp, pairs=pairs, null_test=not a.no_null_test, draws=a.draws, seeds=range(a.seeds), unit=a.unit, gamma=a.gamma)
    json.dump(rep, open(a.out + ".json", "w"), indent=1, default=float)
    open(a.out + ".md", "w").write(report_markdown(rep, os.path.basename(a.edges)))
    print(report_markdown(rep, os.path.basename(a.edges)))


if __name__ == "__main__":
    main()
