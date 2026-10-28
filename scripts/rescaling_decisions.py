#!/usr/bin/env python3
"""Decisions that a rescaling of mu_E changes (Section 8.1).

Reads results/rescaling_validity.json. A decision is the order of two cuts of
one planted graph. Counts the pairs whose order under the Q_F drop on G^phi
differs from the order on the original scale, the pairs ordered differently by
the index and by the Q_F drop, and, for each disagreement, whether the two cuts
have shortfalls within 0.01 of each other, and, for the untied ones, whether the
reversed order (under phi, or the index) agrees with the shortfall. Writes results/rescaling_decisions.json.
"""
import itertools, json

d = json.load(open("results/rescaling_validity.json"))
out = {k: {"pairs": 0, "reversed": 0, "reversed_shortfall_tied": 0, "reversed_k_agrees_with_shortfall": 0} for k in ("drop_cube", "drop_cube_root", "drop_upper", "index")}
for r in d["rows"]:
    c = r["cuts"]
    for a, b in itertools.combinations(c, 2):
        ds = c[a]["shortfall"] - c[b]["shortfall"]; dd = c[a]["drop_identity"] - c[b]["drop_identity"]
        for k in out:
            out[k]["pairs"] += 1
            if (c[a][k] - c[b][k]) * dd < 0:
                out[k]["reversed"] += 1
                out[k]["reversed_shortfall_tied"] += int(abs(ds) < 0.01)
                out[k]["reversed_k_agrees_with_shortfall"] += int(abs(ds) >= 0.01 and (c[a][k] - c[b][k]) * ds > 0)
print(json.dumps(out, indent=1))
json.dump(out, open("results/rescaling_decisions.json", "w"), indent=1)
