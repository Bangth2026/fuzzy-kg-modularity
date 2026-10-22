#!/usr/bin/env python3
"""Section 8.5 / Table 10: interval type-2 state versus the type-1 collapse.

Runs entirely offline on the 201 x 7 = 1407-state grid.  Prints

* embeddedness of the collapse in its own footprint (paper: 92.7% for par);
* mean footprint widths (paper: 0.233 / 0.362 / 0.230);
* the closed-form degeneration half-width h* (paper: 0.075);
* Table 10 in two variants -- the as-published gate (bound vectors normalised
  separately before gating) and the gate on the raw bounds of Eq. (13);
* timing of collapse vs footprint vs footprint + Nie-Tan.

Usage:  python scripts/reproduce_section85.py [--tau 0.4]
"""

import argparse
import time

import numpy as np

from fuzzykg.fuzzification import (
    DEFAULT_CUTS,
    LABELS,
    collapse_type1,
    degeneration_halfwidth,
    embedded_in_footprint,
    footprint,
    nie_tan,
    zpd_gate_it2,
    zpd_gate_it2_normalised,
    zpd_gate_type1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--points", type=int, default=201)
    args = ap.parse_args()
    tau = args.tau

    ps = np.linspace(0.0, 1.0, args.points)
    hs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    P, H = np.meshgrid(ps, hs, indexing="ij")
    print(f"grid: {P.size} states ({args.points} point estimates x {len(hs)} widths), cuts {DEFAULT_CUTS}")

    fou = footprint(P, H)
    ok = all(np.all(fou[k][0] <= fou[k][1] + 1e-12) for k in LABELS)
    inrange = all((fou[k][0].min() >= 0) and (fou[k][1].max() <= 1) for k in LABELS)
    print(f"footprint well formed: lower<=upper {ok}, in [0,1] {inrange}, point at h=0: "
          f"{np.all(fou['par'][1][:, 0] == fou['par'][0][:, 0])}")
    for k in LABELS:
        print(f"  mean footprint width {k:3s}: {(fou[k][1] - fou[k][0]).mean():.3f}")

    emb = embedded_in_footprint(P, H)
    for k in LABELS:
        print(f"  collapse embedded in footprint ({k}): {emb[k].mean() * 100:.1f}%")
    print(f"h* = (1 - tau) max(a, b) / 2 = {degeneration_halfwidth(tau):.3f}")

    # ---------------------------------------------------------------- Table 10
    for variant in ("as published (bounds normalised separately)", "raw bounds, Eq. (13) as stated"):
        print(f"\nTable 10 -- {variant}; tau_ZPD = {tau}")
        print(f"{'eta*sigma':>9} {'IT2 pass':>9} {'IT2 fail':>9} {'IT2 indet':>10} {'T1 pass':>8} {'T1~NieTan':>10}  {'sum':>6}")
        tot = {"pass": 0, "fail": 0, "ind": 0, "t1": 0, "nt": 0, "agree": 0, "n": 0, "t1_fail_on_ind": 0, "n_ind": 0}
        for h in hs:
            t1 = zpd_gate_type1(ps, h, tau)
            f = footprint(ps, h)
            nt = nie_tan(f, renormalise=True)["par"] >= tau
            if variant.startswith("as published"):
                pa, fa, ind = zpd_gate_it2_normalised(ps, h, tau)
            else:
                g = zpd_gate_it2(ps, h, tau)
                pa, fa, ind = g == 1, g == 0, g == -1
            s = pa.mean() + fa.mean() + ind.mean()
            print(f"{h:9.2f} {pa.mean()*100:8.1f}% {fa.mean()*100:8.1f}% {ind.mean()*100:9.1f}% "
                  f"{t1.mean()*100:7.1f}% {np.mean(nt == t1)*100:9.1f}%  {s*100:5.1f}%")
            tot["pass"] += pa.sum(); tot["fail"] += fa.sum(); tot["ind"] += ind.sum()
            tot["t1"] += t1.sum(); tot["nt"] += nt.sum(); tot["agree"] += np.sum(nt == t1); tot["n"] += ps.size
            tot["t1_fail_on_ind"] += np.sum(~t1[ind]); tot["n_ind"] += ind.sum()
        n = tot["n"]
        print(f"  aggregate: collapse passes {tot['t1']/n*100:.1f}%, IT2 passes {tot['pass']/n*100:.1f}%, "
              f"indeterminate {tot['ind']/n*100:.1f}%, Nie-Tan passes {tot['nt']/n*100:.1f}%, "
              f"Nie-Tan agrees with collapse {tot['agree']/n*100:.1f}%; "
              f"of indeterminate states the collapse fails {tot['t1_fail_on_ind']/max(1,tot['n_ind'])*100:.0f}%")

    # ------------------------------------------------------------------ timing
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        collapse_type1(P, H)
    t_c = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        footprint(P, H)
    t_f = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        nie_tan(footprint(P, H))
    t_n = (time.perf_counter() - t0) / reps
    print(f"\ncost over {P.size} states (vectorised, mean of {reps}): collapse {t_c*1e3:.3f} ms, "
          f"footprint {t_f*1e3:.3f} ms ({t_f/t_c:.2f}x), footprint+Nie-Tan {t_n*1e3:.3f} ms ({t_n/t_c:.2f}x)")


if __name__ == "__main__":
    main()
