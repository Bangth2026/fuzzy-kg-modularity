#!/usr/bin/env python3
"""Release 5.2.0. kappa* in the refined family of Corollary 1, searched again.

Release 5.1.0 searched the refined family (J = 12 segments) without enumerating the vertices of the
box, because fuzzykg.simplex._box_extreme enumerated vertices only for J <= 10; the search then rested
on random points and local refinement and returned a larger radius than the family it contains on 6
of the 69 real cuts. With vertex enumeration up to J = 12 the nesting kappa*(refined) <= kappa*(family 5)
is checked on every cut. Writes results/v17/refined.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from fuzzykg.simplex import CutFamily, kappa_star  # noqa: E402
from v15_partitions import GRAPHS, load  # noqa: E402
from v15_simplex_real import group  # noqa: E402
from v16_semisynthetic import refined_family  # noqa: E402

DELTA = json.load(open("results/v15/summary.json"))["delta"]
CUTS = ("q0.25", "q0.50", "q0.75")


def main():
    res = {}
    for name in GRAPHS:
        st = np.load(f"results/v15/partitions/{name}.npz"); G = load(name)
        parts = {w: [st[f"{w}|1|{s}"] for s in range(20)] for w in ("support", *CUTS)}
        dl = DELTA["dense" if group(name) == "co-practice" else "sparse"]
        fam = CutFamily(G, parts); famr = refined_family(G, parts)
        res[name] = {}
        for c in CUTS:
            k0 = kappa_star(fam, c, min_excess=dl, kmax=1000.0); k1 = kappa_star(famr, c, min_excess=dl, kmax=1000.0)
            res[name][c] = {"decision": k0["decision_mu"], "kappa": k0["kappa_star"], "kappa_refined": k1["kappa_star"], "J_refined": famr.J}
        print(name, {c: (round(min(r["kappa"], 1e3), 2), round(min(r["kappa_refined"], 1e3), 2)) for c, r in res[name].items()}, flush=True)
    viol = [(n, c, r["kappa"], r["kappa_refined"]) for n, v in res.items() for c, r in v.items() if min(r["kappa_refined"], 1e3) > min(r["kappa"], 1e3) * 1.02 + 1e-9]
    changed = [(n, c) for n, v in res.items() for c, r in v.items() if (r["kappa"] >= 2) != (r["kappa_refined"] >= 2)]
    json.dump({"cuts": res, "nesting_violations": viol, "robust_changed": changed}, open("results/v17/refined.json", "w"), indent=1, default=float)
    print("violations", viol, "changed", changed)


if __name__ == "__main__":
    main()
