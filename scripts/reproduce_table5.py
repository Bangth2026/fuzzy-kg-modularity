#!/usr/bin/env python3
"""Section 8.2 / Table 5: Fuzzy Leiden on synthetic fuzzy prerequisite graphs.

Offline.  For each regime (p_within = 0.85 strongly modular, 0.55 weakly
modular) and size, runs Fuzzy Leiden at gamma = 0.5 on 5 seeds, recording
runtime, Q_F, its seed-to-seed sd, number of communities, largest community
share, the Proposition 1 check and the Proposition 2(iii) check over 12
optimisation iterations.

Usage:
  python scripts/reproduce_table5.py                 # sizes 59 .. 5000 (a few minutes)
  python scripts/reproduce_table5.py --full          # adds 10000 and 20000
  python scripts/reproduce_table5.py --sizes 59 250  # custom
"""

import argparse
import time

import numpy as np

from fuzzykg.leiden import fuzzy_leiden
from fuzzykg.synthetic import synthetic_fuzzy_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[59, 250, 1000, 5000])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    sizes = args.sizes + ([10000, 20000] if args.full else [])

    print(f"{'p_within':>8} {'|C|':>6} {'|E|':>6} {'time(s)':>8} {'Q_F':>6} {'sd':>6} {'comms':>6} {'max%':>6} {'conn':>5} {'mono':>5}")
    total_comms = 0
    all_diffs = []
    for p_within in (0.85, 0.55):
        for n in sizes:
            QF, times, comms, shares, conn, mono, m_edges = [], [], [], [], 0, 0, []
            for seed in range(args.seeds):
                G = synthetic_fuzzy_graph(n, p_within, seed=seed)
                t0 = time.perf_counter()
                r = fuzzy_leiden(G, args.gamma, seed=seed, n_iterations=args.iters)
                times.append(time.perf_counter() - t0)
                QF.append(r.Q_F); comms.append(r.n_communities); shares.append(max(r.sizes) / n)
                conn += r.connected_violations; mono += r.monotone_violations
                m_edges.append(G.m); total_comms += r.n_communities
                all_diffs.extend(np.diff(r.trajectory[: args.iters + 1]).tolist())
            print(f"{p_within:8.2f} {n:6d} {int(np.mean(m_edges)):6d} {np.mean(times):8.3f} {np.mean(QF):6.3f} "
                  f"{np.std(QF, ddof=0):6.3f} {np.mean(comms):6.1f} {np.mean(shares)*100:5.1f}% "
                  f"{'pass' if conn == 0 else conn:>5} {'pass' if mono == 0 else mono:>5}")
    print(f"\ncommunities checked for Proposition 1: {total_comms}; "
          f"successive Q_F differences checked for Proposition 2(iii): {len(all_diffs)}, "
          f"minimum {min(all_diffs):.2e}")


if __name__ == "__main__":
    main()
