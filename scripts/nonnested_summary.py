#!/usr/bin/env python3
"""Within-graph concordance for the equal-size binarisations of
nonnested_binarisation.py (Section 8.1). A lower retained mass predicts a larger
shortfall, so its sign is flipped; the mean out-degree is identical across rules
by construction (concordance 0.5). Reported over all four rules and over the
three informed rules (global, per-source, disparity), whose retained masses
differ by a few hundredths. Pairs whose shortfalls differ by less than 0.01 are
skipped. Writes results/nonnested_summary.json.
"""
import itertools, json
import numpy as np

rows = json.load(open("results/nonnested_binarisation.json"))["rows"]
rng = np.random.default_rng(0)


def conc(key, sign, rules):
    v, npairs = [], 0
    for r in rows:
        c = {k: x for k, x in r["cells"].items() if rules is None or k in rules}
        s = n = 0
        for a, b in itertools.combinations(c, 2):
            dl = c[a]["shortfall"] - c[b]["shortfall"]
            if abs(dl) < 0.01:
                continue
            n += 1; dd = sign * (c[a][key] - c[b][key])
            s += 1 if dl * dd > 0 else (0.5 if dd == 0 else 0)
        if n:
            v.append(s / n); npairs += n
    v = np.array(v); b = [rng.choice(v, v.size).mean() for _ in range(2000)]
    return {"concordance": float(v.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "graphs": int(v.size), "pairs": npairs}


out = {}
for name, rules in (("all_rules", None), ("informed_rules", ("global", "per_source", "disparity"))):
    out[name] = {k: conc(k, sg, rules) for k, sg in (("index", 1), ("qf_drop", 1), ("retained_mass", -1), ("singleton_share", 1))}
    sel = [r["cells"][k]["retained_mass"] for r in rows for k in (rules or r["cells"])]
out["mass_range_informed"] = float(np.mean([max(r["cells"][k]["retained_mass"] for k in ("global", "per_source", "disparity")) -
                                            min(r["cells"][k]["retained_mass"] for k in ("global", "per_source", "disparity")) for r in rows]))
print(json.dumps(out, indent=1))
json.dump(out, open("results/nonnested_summary.json", "w"), indent=1)
