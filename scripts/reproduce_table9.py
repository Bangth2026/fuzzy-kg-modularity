#!/usr/bin/env python3
"""Section 8.4 / Table 9: verification of the fuzzification and cut-point sensitivity.

Offline.  505-point grid (101 point estimates x 5 widths), each cut-point
perturbed over {-0.10, -0.05, 0, +0.05, +0.10} (125 joint settings, 63125
fuzzifications).  Checks sum-to-one, range and monotonicity, and reports the
L1 distance to the baseline triple and the defuzzified-band flip rate.

Usage:  python scripts/reproduce_table9.py
"""

import itertools

import numpy as np

from fuzzykg.fuzzification import DEFAULT_CUTS, collapse_type1

ps = np.linspace(0.0, 1.0, 101)
hs = [0.0, 0.05, 0.10, 0.20, 0.30]
P, H = np.meshgrid(ps, hs, indexing="ij")


def triple(cuts):
    return np.stack(collapse_type1(P, H, cuts))


def properties(T):
    u, r, m = T
    sum_to_one = np.abs(T.sum(0) - 1).max() < 1e-5
    in_range = (T.min() >= 0) and (T.max() <= 1)
    mono = bool(np.all(np.diff(m, axis=0) >= -1e-12) and np.all(np.diff(u, axis=0) <= 1e-12))
    return sum_to_one, in_range, mono


def main():
    base = triple(DEFAULT_CUTS)
    print(f"grid: {P.size} states; baseline cuts {DEFAULT_CUTS}")
    s, r, m = properties(base)
    print(f"baseline: sum-to-one {s}, range {r}, monotone {m}")
    for p, expect in ((0.95, (0, 0, 1)), (0.05, (1, 0, 0)), (0.60, (0, 1, 0))):
        print(f"  p*={p:.2f}, h=0 -> {tuple(round(float(x), 6) for x in collapse_type1(p, 0.0))}  (expected {expect})")

    print(f"\n{'Perturbation':16s} {'Mean L1':>8} {'Max L1':>8} {'Band flips':>11}  props")
    names = ("a_unk", "a_par", "a_mas")
    for i, nm in enumerate(names):
        for d in (-0.10, -0.05, 0.05, 0.10):
            c = list(DEFAULT_CUTS); c[i] = round(c[i] + d, 10)
            T = triple(tuple(c))
            l1 = np.abs(T - base).sum(0)
            flips = (T.argmax(0) != base.argmax(0)).mean()
            print(f"{nm + f' {d:+.2f}':16s} {l1.mean():8.3f} {l1.max():8.3f} {flips*100:10.1f}%  {properties(T)}")

    for label, exact in (("as published: 125 settings, raw floating-point cut-points", False),
                         ("strictly ordered settings only (exact arithmetic)", True)):
        rows = []
        skipped = 0
        for du, dp, dm in itertools.product((-0.1, -0.05, 0.0, 0.05, 0.1), repeat=3):
            if exact:
                c = (round(0.35 + du, 10), round(0.60 + dp, 10), round(0.80 + dm, 10))
            else:
                c = (0.35 + du, 0.60 + dp, 0.80 + dm)
            if not (c[0] < c[1] < c[2]):
                skipped += 1
                continue
            T = triple(c)
            l1 = np.abs(T - base).sum(0)
            rows.append((l1.mean(), (T.argmax(0) != base.argmax(0)).mean(), properties(T)))
        print(f"\njoint perturbation, {label}: {len(rows)} settings evaluated, {skipped} skipped")
        print(f"  mean of settings: L1 {np.mean([x[0] for x in rows]):.3f}, flips {np.mean([x[1] for x in rows])*100:.1f}%")
        print(f"  worst setting:    L1 {max(x[0] for x in rows):.3f}, flips {max(x[1] for x in rows)*100:.1f}%")
        print(f"  all properties hold in every setting: {all(all(x[2]) for x in rows)}")
    print("\nnote: in exact arithmetic the five settings with (a_par + 0.10, a_mas - 0.10) have a_par = a_mas = 0.70,"
          " a zero-width mastered band.  In floating point 0.60 + 0.10 > 0.80 - 0.10 by one ulp, which is why the"
          " published run counted 125 well-ordered settings; the mastered transition is a step there.")


if __name__ == "__main__":
    main()
