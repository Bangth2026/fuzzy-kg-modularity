"""Gates on the uncertain state: the cautious collapse against four alternative
gates (Supplementary Table S5), on the same 201 x 7 grid as Supplementary Section S1.6.

  collapse      Eq. (11)-(12), pass iff collapsed mu_par >= tau
  point         (i)  m_par(p) >= tau, dispersion ignored
  interval-T1   (ii) min_{[lo,hi]} m_par >= tau  (= the IT2 'pass' rule)
  prob-ZPD      (iii) P(m_par(X) >= tau) >= 1/2 under X ~ N(p, sigma^2)
  expected      (iv) mean_{[lo,hi]} m_par >= tau
  IT2           three-valued: pass / indeterminate / fail

Prints the pass rate of each gate per half-width and the agreement of each
baseline with the IT2 'pass' and 'fail' outcomes.  Usage: python scripts/gate_baselines.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg.fuzzification import (zpd_gate_type1, zpd_gate_it2, gate_point_estimate,
    gate_interval_image_type1, gate_probability, gate_expected_membership)

def main(tau=0.4, out="results/gate_baselines.json"):
    ps = np.linspace(0, 1, 201)
    widths = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows = []
    for h in widths:
        H = np.full_like(ps, h)
        it2 = zpd_gate_it2(ps, H, tau)
        gates = {"collapse": zpd_gate_type1(ps, H, tau), "point": gate_point_estimate(ps, H, tau),
                 "interval_T1": gate_interval_image_type1(ps, H, tau),
                 "prob_ZPD": gate_probability(ps, H, tau), "expected": gate_expected_membership(ps, H, tau)}
        row = {"halfwidth": h, "IT2_pass": float((it2 == 1).mean()), "IT2_indet": float((it2 == -1).mean()),
               "IT2_fail": float((it2 == 0).mean())}
        for k, g in gates.items():
            row[f"{k}_pass"] = float(g.mean())
            # does the baseline pass something the IT2 gate calls fail, or reject an IT2 pass?
            row[f"{k}_passes_IT2fail"] = float((g & (it2 == 0)).mean())
            row[f"{k}_rejects_IT2pass"] = float((~g & (it2 == 1)).mean())
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
