#!/usr/bin/env python3
"""MOOCCubeX-CS prerequisite graph (Section 8, Table 2 input).

Reads data/raw/MOOCCubeX/prerequisites/cs.json and relations/concept-problem.txt,
keeps edges whose endpoints both carry at least one graded problem, and uses the
released confidences as mu_E.  Writes data/graphs/mooccubex_cs_{edges,nodes}.csv.

Usage:  python scripts/build_mooccubex_graph.py [--root data/raw/MOOCCubeX]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import load_mooccubex_cs  # noqa: E402
from fuzzykg import FuzzyGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "graphs"))
    ap.add_argument("--full", action="store_true", help="keep all ground-truth pairs (no assessed-concept restriction)")
    args = ap.parse_args()
    edges, assessed = load_mooccubex_cs(args.root, full=args.full)
    name = "mooccubex_cs_full" if args.full else "mooccubex_cs"
    labels = sorted(set(edges.src) | set(edges.dst))
    idx = {k: i for i, k in enumerate(labels)}
    G = FuzzyGraph.from_edges(((idx[a], idx[b], m) for a, b, m in zip(edges.src, edges.dst, edges.mu)),
                              n=len(labels), labels=labels)
    print(f"assessed concepts: {len(assessed)}; graph: {G.summary()}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"src": G.src, "dst": G.dst, "mu": G.mu}).to_csv(out / f"{name}_edges.csv", index=False)
    pd.DataFrame({"id": range(G.n), "label": G.labels}).to_csv(out / f"{name}_nodes.csv", index=False)
    print(f"wrote {out}/{name}_{{edges,nodes}}.csv")


if __name__ == "__main__":
    main()
