#!/usr/bin/env python3
"""Release 5.2.0. Markdown of the new and replaced Supplementary tables -> results/v17/supp_tables.json."""
from __future__ import annotations
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from v16_supp_tables import f, k, table, NAMES  # noqa: E402

S = json.load(open("results/v17/summary.json"))
I16 = json.load(open("results/v16/summary.json"))["injected"]
NAMES = dict(NAMES, junyi_transition_unsuggested="Junyi transition, unsuggested")


def ba(m):
    return "n/a" if m["ba"] is None else f(m["ba"], 2)


def s2():
    R = S["refined"]
    return ("*The refined family (Corollary 1).* With a knot added just below each of the three cut levels (J = 12 instead of 9), κ\\* was searched again for all 69 real cuts at the class δ, "
            "with every vertex of each box enumerated, as for the family (5). The refined family contains the family (5), so its radius can only be smaller or equal; "
            f"the search respects this on all 69 cuts ({len(R['nesting_violations'])} violations), the radii agree to within 1% on every cut, and no robust decision changes. "
            "(An earlier search of the refined family that enumerated vertices only up to J = 10 relied on random and local points in twelve dimensions and returned larger radii on six cuts.)")


def s3h():
    out = []
    rows = []
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        c = S["classes"][cls]
        for e, a in c["audit"].items():
            x = I16[f"{con}|{e}"]
            rows.append([con, e, f(x["V_mean"], 3, True), f(x["share_V_pos"], 2),
                         f"{a['no_check_vs_support']['losses']} / {a['no_check_vs_support']['no_loss']}", ba(a["no_check_vs_support"]),
                         f"{a['no_check_vs_F']['losses']} / {a['no_check_vs_F']['no_loss']}", ba(a["no_check_vs_F"]),
                         f"{a['no_check_vs_F']['cut_right']} of {a['no_check_vs_F']['cut_decisions']}"])
    out.append("**S3h. Grades that mark the communities.** The memberships of every graph of the 96 worlds transformed after construction, μ′ = μ^(1/(1+η)) inside and μ^(1+η) between true communities. The transformation also changes the quantile cuts, which keep internal edges preferentially, so at η \\> 0 cuts lose less against the support. V: mean partition value; V \\> 0.02: share of worlds; losses / no losses among the cuts the robust rule decides, and its balanced accuracy, against the support and against Π_F (n/a: one class empty); robust cut decisions right against Π_F.\n\n" +
               table(["Construction", "η", "V", "V \\> 0.02", "Loss / none vs support", "BA vs support", "Loss / none vs Π_F", "BA vs Π_F", "Cuts right vs Π_F"], rows))
    rows = []
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        c = S["classes"][cls]
        for tag in ("plain", "null"):
            t = c[f"test_{tag}"]; name = c[f"selected_{tag}"]
            rows.append([cls, "plain D" if tag == "plain" else "null-referenced D", name.replace("Dnull", "D_null").replace("null_min", "below every null"),
                         f(t["sens"], 2), f(t["false"], 2), f(t["ba"], 2)])
    out.append("**S3i. The divergence check by density class.** Thresholds chosen on the calibration worlds (seed 0) by balanced accuracy for a partition value above 0.02, from D ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} for the plain divergence (7) and D_null ∈ {−0.02, 0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2} or 'below every null' for the null-referenced one (ten row permutations); sensitivity, false-alarm rate and balanced accuracy on the test worlds (seed 1), all η pooled.\n\n" +
               table(["Graphs", "Check", "Selected rule", "Sensitivity", "False alarm", "BA"], rows))
    rows = []
    for con, cls in (("co-practice", "dense"), ("transition", "sparse")):
        c = S["classes"][cls]
        for e, a in c["audit"].items():
            rows.append([cls, e, ba(a["no_check_vs_F"]), f"{a['no_check_vs_F']['cut_right']} of {a['no_check_vs_F']['cut_decisions']}",
                         ba(a["plain_vs_F"]), f"{a['plain_vs_F']['cut_right']} of {a['plain_vs_F']['cut_decisions']}",
                         ba(a["null_vs_F"]), f"{a['null_vs_F']['cut_right']} of {a['null_vs_F']['cut_decisions']}"])
    out.append("**S3j. The audit against Π_F with the checks.** Every cut of a graph on which the check fires is flagged; balanced accuracy and right cut decisions against the loss relative to Π_F, all worlds, without a check and with the plain and null-referenced checks at the thresholds of Table S3i.\n\n" +
               table(["Graphs", "η", "BA, no check", "Cuts right", "BA, plain D", "Cuts right", "BA, null D", "Cuts right"], rows))
    return "\n\n".join(out)


def s4fg():
    B = S["misc"]["block_null"]; T = S["misc"]["tau_crossfit"]
    rows = [[NAMES[n], f(b["support"]), f(b["observed"]), f(b["global"]), f(b["row"]), f(b["block"]), f(b["V_total"], 3, True), f(b["V_beyond_block"], 3, True),
             f(b["mu_internal_mean"]), f(b["mu_between_mean"]), f(b["share_internal_edges"], 2)] for n, b in B.items()]
    o = ["**Table S4f.** Mechanism of the two positive partition values. ARI with the grouping (best of three seeds; nulls: ten draws, best of three) for the support, the graded graph, the global and row nulls and a grouping-class null that permutes the memberships separately among the edges inside one group and among the edges between groups; V; the value beyond the grouping-class null; mean membership of edges inside and between groups and the share of edges inside a group. The unsuggested graph is Junyi transition rebuilt from the attempts the platform did not suggest.\n\n" +
         table(["Graph", "Support", "Graded", "Global", "Row", "Group-class", "V", "V beyond group-class", "μ inside", "μ between", "Inside share"], rows)]
    P = T["point"]
    rows = [[t if t != "0" else "none", f(P[f"hop|{t}"]), f(P[f"product|{t}"])] for t in ("0", "0.05", "0.1", "0.2", "0.3")]
    ch = ", ".join(f"crisp {a.split('|')[0]} / graded {('none' if a.split('|')[1] == '0' else a.split('|')[1])}: {v:.2f}" for a, v in sorted(T["choices"].items(), key=lambda x: -x[1]))
    o.append("**Table S4g.** Best crisp against best graded representation of the Junyi transition graph over the pruning threshold (fixed 1,954 annotated pairs): Spearman ρ of the hop score 1/(1 + d) and of the product best-path score at each threshold. "
             f"Cross-fitted comparison, both thresholds chosen on the pairs inside a random half of the exercises and evaluated on the pairs inside the other half (200 halvings): held-out difference graded − crisp {f(T['heldout_diff'][0], 3, True)} \\[{f(T['heldout_diff'][1], 3, True)}, {f(T['heldout_diff'][2], 3, True)}\\], positive in {T['share_positive']:.0%} of the halvings; thresholds chosen ({ch}).\n\n" +
             table(["τ_E", "ρ hop (crisp)", "ρ product (graded)"], rows))
    return "\n\n".join(o)


def s5c():
    R = S["real"]; G16 = json.load(open("results/v16/real.json"))["graphs"]
    rows = []
    for n, d in R.items():
        g = G16[n]["gate"]
        rows.append([NAMES[n], d["class"], f(d["a_FS"]), f(d["a_SS"]), f(d["a_FF"]), f(d["D"], 3, True), "yes" if (d["class"] == "dense" and d["fires_plain_class"]) else ("--" if d["class"] == "sparse" else "no"),
                     f(d["A_null_mean"]), f(d["D_null"], 3, True), f(g["g_obs"], 4), f(g["null_max"], 4)])
    return ("**Table S5c.** The checks of Section 4.5 on the 23 graphs: agreement of Π_F with Π_sup (a_FS) and of consecutive seeds of each (20 seeds); divergence D of (7) and whether the check fires (dense graphs, D0 = 0.02; not run on sparse graphs); mean agreement of the partitions on ten row permutations of μ_E with Π_sup and the null-referenced divergence D_null; the gain g of Π_F over Π_sup on μ_E and its largest value over the row permutations.\n\n" +
            table(["Graph", "Class", "a_FS", "a_SS", "a_FF", "D", "Check fires", "Row-null agreement", "D_null", "Gain g", "Max over row nulls"], rows))


def s7c():
    J = S["junyi_fraction"]; L = json.load(open("results/v16/learner_resampling.json"))["junyi_transition"]
    rows = []
    for frac, draws in sorted(J.items(), key=lambda x: float(x[0])):
        rows.append([f"{float(frac):.0%}", str(len(draws)), f"{np.median([r['m'] for r in draws]):.0f}"] + [f(float(np.median([min(r['audit'][c]['kappa_star'], 1e3) for r in draws])), 2) for c in ("q0.25", "q0.50", "q0.75")])
    s = L["summary"]
    rows.append(["80%", str(len(L["subsample"])), f"{s['q0.25']['subsample']['edges_median']:.0f}"] + [f(s[c]["subsample"]["kappa_star_quartiles"][1], 2) for c in ("q0.25", "q0.50", "q0.75")])
    rows.append(["100%", "1", str(L["observed"]["m"])] + [f(L["observed"]["audit"][c]["kappa_star"], 2) for c in ("q0.25", "q0.50", "q0.75")])
    return ("**Table S7c.** κ\\* of the Junyi transition cuts against the share of learners kept (subsamples without replacement; ten seeds, sparse δ): median number of edges and median κ\\* per cut. Smaller samples give sparser graphs and larger radii.\n\n" +
            table(["Learners kept", "Draws", "Edges", "κ\\* q = 0.25", "κ\\* q = 0.50", "κ\\* q = 0.75"], rows))


if __name__ == "__main__":
    out = {"S2": s2(), "S3h": s3h(), "S4fg": s4fg(), "S5c": s5c(), "S7c": s7c()}
    json.dump(out, open("results/v17/supp_tables.json", "w"), indent=1, ensure_ascii=False)
    for kk, v in out.items():
        print(kk, len(v))
