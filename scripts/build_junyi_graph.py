#!/usr/bin/env python3
"""Build the two Junyi Academy fuzzy graphs (co-practice, Section 7.1; order-preserving
transition construction) from the public problem log, and write them in the format of
data/graphs/.  Usage: python scripts/build_junyi_graph.py [--construction copractice|transition|both]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_junyi  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_fuzzy_graph, build_transition_graph, preprocess  # noqa: E402


def write(G, tab, name, extra):
    os.makedirs("data/graphs", exist_ok=True)
    pd.DataFrame({"src": G.src, "dst": G.dst, "mu": G.mu}).to_csv(f"data/graphs/{name}_edges.csv", index=False)
    pd.DataFrame({"id": range(G.n), "label": G.labels}).to_csv(f"data/graphs/{name}_nodes.csv", index=False)
    tab.to_csv(f"data/graphs/{name}_signals.csv", index=False)
    print(name, G.summary(), json.dumps(extra), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", default="both")
    args = ap.parse_args()
    t0 = time.perf_counter()
    df = load_junyi()
    d = preprocess(df, ConstructionConfig())
    info = {"learners": int(d.user_id.nunique()), "interactions": int(len(d)), "load_s": round(time.perf_counter() - t0, 1)}
    if args.construction in ("copractice", "both"):
        t = time.perf_counter(); G, tab = build_fuzzy_graph(d, ConstructionConfig(), preprocessed=True)
        write(G, tab, "junyi_copractice", dict(info, build_s=round(time.perf_counter() - t, 1)))
    if args.construction in ("transition", "both"):
        t = time.perf_counter(); G, tab = build_transition_graph(d, TransitionConfig(), preprocessed=True)
        write(G, tab, "junyi_transition", dict(info, build_s=round(time.perf_counter() - t, 1)))


if __name__ == "__main__":
    main()
