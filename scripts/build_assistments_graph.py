#!/usr/bin/env python3
"""Section 7.1 end to end on an ASSISTments release (Table 3).

Usage:
  python scripts/build_assistments_graph.py --release 2009 --level kc
  python scripts/build_assistments_graph.py --release 2009 --level item
  python scripts/build_assistments_graph.py --release 2015 --level kc
  python scripts/build_assistments_graph.py --release 2017 --level kc
  python scripts/build_assistments_graph.py --release 2017 --level item

Defaults are those of Section 8.1: horizon 10, minimum support 20,
OWA w = (0.5, 0.3, 0.2), no taxonomy prior, tau_E = 0.2, logistic slope 5,
learners with < 15 interactions dropped.  Writes data/graphs/assist<release>_<level>_{edges,nodes,signals}.csv.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import load_assistments  # noqa: E402
from fuzzykg.construction import ConstructionConfig, TransitionConfig, build_fuzzy_graph, build_transition_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, choices=["2009", "2015", "2017"])
    ap.add_argument("--level", default="kc", choices=["kc", "item"])
    ap.add_argument("--path", default=None, help="override the raw CSV path")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--tau-e", type=float, default=0.2)
    ap.add_argument("--orness", type=float, default=None, help="regenerate OWA weights from an orness target (Section 8.4)")
    ap.add_argument("--construction", default="copractice", choices=["copractice", "transition"],
                    help="co-practice construction of Section 7.1 or the order-preserving transition construction")
    ap.add_argument("--out", default=str(ROOT / "data" / "graphs"))
    for c in ("user", "kc", "correct", "order"):
        ap.add_argument(f"--{c}", default=None, help=f"column override for {c}")
    args = ap.parse_args()

    cols = {c: getattr(args, c) for c in ("user", "kc", "correct", "order") if getattr(args, c)}
    df = load_assistments(args.release, args.level, args.path, **cols)
    print(f"loaded {len(df)} rows, {df.user_id.nunique()} learners, {df.kc.nunique()} {args.level} nodes")

    cfg = ConstructionConfig(horizon=args.horizon, min_support=args.min_support, tau_E=args.tau_e)
    if args.orness is not None:
        from fuzzykg.tnorms import owa_weights_from_orness
        cfg.owa_weights = tuple(owa_weights_from_orness(args.orness, 3))
        print(f"OWA weights from orness {args.orness}: {cfg.owa_weights}")
    t0 = time.perf_counter()
    if args.construction == "copractice":
        G, tab = build_fuzzy_graph(df, cfg)
    else:
        G, tab = build_transition_graph(df, TransitionConfig(min_support=args.min_support, tau_E=args.tau_e))
    dt = time.perf_counter() - t0
    print(G.summary(), f"build {dt:.1f} s")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    name = f"assist{args.release}_{args.level}" + ("" if args.construction == "copractice" else "_transition")
    import pandas as pd
    pd.DataFrame({"src": G.src, "dst": G.dst, "mu": G.mu}).to_csv(out / f"{name}_edges.csv", index=False)
    pd.DataFrame({"id": range(G.n), "label": G.labels}).to_csv(out / f"{name}_nodes.csv", index=False)
    tab.to_csv(out / f"{name}_signals.csv", index=False)
    print(f"wrote {out / name}_{{edges,nodes,signals}}.csv")


if __name__ == "__main__":
    main()
