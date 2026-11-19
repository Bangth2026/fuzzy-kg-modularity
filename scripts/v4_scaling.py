#!/usr/bin/env python3
"""Version 4.0.0. Run time and peak memory of every step of Algorithms 1 and 2 against the
number of nodes and edges (reply to the request for computational cost).

Synthetic directed planted-partition fuzzy graphs (fuzzykg.synthetic, 20 communities, p_in 0.7,
memberships Beta with mean 0.5 and SD 0.15) at n in {1000, 2000, 5000, 10000, 20000} and mean
out-degree in {8, 32}. Each operation runs in a fresh subprocess; reported are the wall time and
the peak resident memory above the memory of the interpreter with the graph loaded.
Operations: one Leiden run on Q_F (= one rank-coded Q_F drop), one multiplex Leiden run over
the eight-level quantile grid (the cost index), one degree-preserving null draw (rewiring and
Leiden), one weights-only null draw (row permutation and Leiden), one planted graph of the
structure-matched floor, and best paths from 200 sources under the product (Dijkstra on the
additive generator) and under max-min (widest path), extrapolated to all sources.
Writes results/v4/scaling.json.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

OPS = ["leiden", "multiplex", "degree_draw", "weights_draw", "dc_planted", "paths_product", "paths_maxmin"]
CODE = r'''
import sys, time, resource, json, numpy as np
sys.path.insert(0, ".")
from fuzzykg.synthetic import planted_partition_fuzzy_graph
from fuzzykg import FuzzyGraph
n, deg, op = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
out = planted_partition_fuzzy_graph(n=n, k=20, out_degree=deg, p_in=0.7, mu_mean=0.5, mu_sd=0.15, seed=0)
G = out[0] if isinstance(out, tuple) else out
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rng = np.random.default_rng(0)
t = time.perf_counter()
if op == "leiden":
    from fuzzykg.nulltest import best_qf; best_qf(G, 1.0, (0,))
elif op == "multiplex":
    from fuzzykg.multislice import multislice_leiden; multislice_leiden(G, seed=0)
elif op == "degree_draw":
    from fuzzykg.nulltest import best_qf, degree_rewire; best_qf(degree_rewire(G, rng), 1.0, (0,))
elif op == "weights_draw":
    from fuzzykg.nulltest import best_qf, permute_rows; best_qf(permute_rows(G, rng), 1.0, (0,))
elif op == "dc_planted":
    from fuzzykg.nulltest import planted_dc_like
    sizes = np.bincount(rng.integers(0, 20, G.n)); planted_dc_like(G, sizes, 0.7, rng)
elif op in ("paths_product", "paths_maxmin"):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    S = min(200, G.n); src = rng.choice(G.n, S, replace=False)
    A = sp.csr_matrix((G.mu, (G.src, G.dst)), shape=(G.n, G.n))
    if op == "paths_product":
        L = A.copy(); L.data = -np.log(L.data); dijkstra(L, directed=True, indices=src)
    else:
        from fuzzykg.paths_fast import _widest_all
        import numba
        # widest paths from a subset: the compiled all-pairs kernel on the sub-problem is not
        # available, so time the full kernel on graphs with n <= 5000 and extrapolate otherwise
        if G.n <= 5000:
            _widest_all(A.indptr, A.indices, A.data, G.n); S = G.n
        else:
            raise SystemExit(json.dumps({"skipped": "dense n x n output"}))
dt = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
r = {"time_s": dt, "peak_extra_MB": (peak - base) / 1024.0, "n": G.n, "m": int(G.m)}
if op.startswith("paths"):
    r["sources_timed"] = int(S); r["time_all_sources_s"] = dt * G.n / S
print(json.dumps(r))
'''


def main():
    out = "results/v4/scaling.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    for deg in (8, 32):
        for n in (1000, 2000, 5000, 10000, 20000):
            for op in OPS:
                key = f"n{n}_d{deg}_{op}"
                if key in res:
                    continue
                if op == "multiplex" and n * deg > 320000:
                    pass
                t0 = time.perf_counter()
                try:
                    p = subprocess.run([sys.executable, "-c", CODE, str(n), str(deg), op], capture_output=True, text=True, timeout=1800)
                    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                    res[key] = json.loads(line[-1]) if line else {"error": p.stderr[-400:]}
                except subprocess.TimeoutExpired:
                    res[key] = {"timeout_s": 1800}
                json.dump(res, open(out, "w"), indent=1)
                print(key, res[key], f"{time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
