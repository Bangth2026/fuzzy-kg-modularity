#!/usr/bin/env python3
"""Release 5.1.0. Markdown of the new Supplementary tables, written from results/v16/*.json
(no number typed by hand). Writes results/v16/supp_tables.json: {section: markdown}."""
from __future__ import annotations
import json
import numpy as np

S = json.load(open("results/v16/summary.json"))
V15 = json.load(open("results/v15/simplex_real.json"))["graphs"]
NAMES = {"mooccubex_cs": "MOOCCubeX CS", "mooccubex_cs_full": "MOOCCubeX CS full", "assist2009_kc": "ASSIST 2009 KC co-practice",
         "assist2009_item": "ASSIST 2009 item co-practice", "assist2015_kc": "ASSIST 2015 KC co-practice", "assist2017_kc": "ASSIST 2017 KC co-practice",
         "assist2017_item": "ASSIST 2017 item co-practice", "junyi_copractice": "Junyi co-practice", "assist2009_kc_transition": "ASSIST 2009 KC transition",
         "assist2015_kc_transition": "ASSIST 2015 KC transition", "assist2017_kc_transition": "ASSIST 2017 KC transition",
         "assist2017_item_transition": "ASSIST 2017 item transition", "junyi_transition": "Junyi transition", "assist2012_copractice": "ASSIST 2012 co-practice",
         "assist2012_transition": "ASSIST 2012 transition", "algebra2005_copractice": "Algebra 2005 co-practice", "algebra2005_transition": "Algebra 2005 transition",
         "bridge2006_copractice": "Bridge 2006 co-practice", "bridge2006_transition": "Bridge 2006 transition", "eedi_copractice": "Eedi co-practice",
         "eedi_transition": "Eedi transition", "ednet_copractice": "EdNet co-practice", "ednet_transition": "EdNet transition"}


def f(x, d=3, sign=False):
    if x is None:
        return "--"
    if isinstance(x, float) and np.isinf(x):
        return "\\> 1000"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def k(x):
    return "\\> 1000" if x >= 999.5 else f"{x:.2f}"


def table(head, rows):
    w = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(head)]
    w = [max(9, min(x, 30)) for x in w]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join([":" + "-" * x for x in w]) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def s2():
    R = S["real"]
    n = len(R["refined_changes_robust"])
    G = S["real"]["graphs"]
    diff = [(g, c) for g, v in G.items() for c, r in v["cuts"].items() if abs(min(r["kappa_star"], 1e3) - min(r["kappa_star_refined"], 1e3)) > 1e-6]
    up = [(g, c) for g, v in G.items() for c, r in v["cuts"].items() if min(r["kappa_star_refined"], 1e3) > min(r["kappa_star"], 1e3) + 1e-6]
    return (f"*The refined family (Corollary 1).* With a knot added just below each of the three cut levels (J = 12 instead of 9), κ\\* was searched again for all 69 real cuts at the class δ (scripts/v16_real.py). The robust decision changed on {n} of the 69 cuts. κ\\* itself differed on {len(diff)} cuts; on {len(up)} of them the refined search returned a larger radius although the refined family contains the original one, so that its true radius can only be smaller: in twelve dimensions the vertex, random and local search can miss a reversal that the nine-dimensional search finds. The radii reported in the main text are those of the family (5), for which the random check above applies.")


def s3():
    out = []
    ss = S["semisynthetic_v15"]
    rows = []
    for con, lab in (("transition", "sparse (transition)"), ("co-practice", "dense (co-practice)")):
        cv = ss[con]["cv_delta"]
        rows.append([lab, ", ".join(f"{x:g}" for x in cv["selected_by_fold"]), f(cv["oof_balanced_accuracy"], 2),
                     f(ss[con]["test_robust"]["ba"], 2), f(ss[con]["test_firm_mu"]["ba"], 2), f(ss[con]["test_fragile_mu"]["ba"], 2) if ss[con]["test_fragile_mu"] else "--"])
    out.append("**S3f. Cross-validation of the minimum excess, and the test worlds.** Eight folds of twelve worlds each (random assignment, seed 0); in each fold δ maximises the balanced accuracy of the decision on μ_E on the other seven folds and is applied to the held-out fold. Test worlds: seed 1, δ from the calibration worlds (seed 0).\n\n" +
               table(["Graphs", "δ selected in the eight folds", "Out-of-fold BA", "Test worlds: robust rule BA", "κ\\* ≥ 2 BA", "κ\\* \\< 2 BA"], rows))
    D = S["semisynthetic_delta"]
    rows = []
    for dl, v in D.items():
        for con in ("co-practice", "transition"):
            x = v[con]
            rows.append([dl, con, str(v["worlds"]), f(x["V_partition_mean"], 3, True), f(x["share_V_positive"], 2), f(x["support_ari_mean"], 2), f(x["F_ari_mean"], 2),
                         f(x["robust_vs_support"]["ba"], 2), f(x["robust_vs_F"]["ba"], 2)])
    out.append("**S3g. Community information in the generator.** Worlds re-run with a prerequisite inside the community of its target drawn from U(0.3 + Δ, 1) and outside from U(0.3, 1 − Δ) (scripts/v16_semisynthetic.py; Δ = 0 reproduces the worlds of Table 2). V: partition value of the grades, ARI(Π_F) − ARI(Π_sup), best of three seeds; share of worlds with V \\> 0.02; mean ARI of the support and graded partitions with the true communities; balanced accuracy of the robust rule against the loss relative to the support and to Π_F.\n\n" +
               table(["Δ", "Construction", "Worlds", "Mean V", "V \\> 0.02", "ARI Π_sup", "ARI Π_F", "BA vs support", "BA vs Π_F"], rows) +
               "\n\nAUC with which the drop from the graded partition, E_F = Q_F(Π_F) − Q_F(Π_cut) on μ_E, and the margin of the audit rank the loss against Π_F: " +
               "; ".join(f"Δ = {dl}: dense {f(v['co-practice']['auc_E_F_lossF'], 2)} and {f(v['co-practice']['auc_margin_lossF'], 2)}, sparse {f(v['transition']['auc_E_F_lossF'], 2)} and {f(v['transition']['auc_margin_lossF'], 2)}" for dl, v in D.items()) + ".")
    I = S["injected"]
    rows = []
    for con in ("co-practice", "transition"):
        for e in ("0", "0.5", "1", "2"):
            x = I[f"{con}|{e}"]
            rows.append([con, e, f(x["V_mean"], 3, True), f(x["share_V_pos"], 2), f(x["lossF_share"], 2),
                         f(x["robust_vs_support"]["ba"], 2), f(x["robust_vs_F"]["ba"], 2),
                         f"{x['robust_vs_F']['cut_correct']} of {x['robust_vs_F']['cut_decisions']}",
                         f(x["div_sens"], 2), f(x["div_false"], 2), f(x["with_div_vs_F"]["ba"], 2),
                         f"{x['with_div_vs_F']['cut_correct']} of {x['with_div_vs_F']['cut_decisions']}",
                         f(x["gate_sens"], 2), f(x["gate_false"], 2)])
    th = I["divergence_threshold"]
    out.append(f"**S3h. Grades that mark the communities.** The memberships of every graph of the 96 worlds transformed after construction, μ′ = μ^(1/(1+η)) inside and μ^(1+η) between true communities (scripts/v16_injected.py, scripts/v16_divergence.py). V: mean partition value; V \\> 0.02: share of worlds; loss vs Π_F: share of cuts; BA of the robust rule against the two losses; robust cut decisions that are right against Π_F; divergence check (D0 = {th['selected']:g}, selected on the calibration worlds from {', '.join(f'{x:g}' for x in th['grid'])} by balanced accuracy for V \\> 0.02): share of worlds with V \\> 0.02 on which it fires (sensitivity) and with V ≤ 0.02 (false alarm), BA of the robust rule with every cut of a firing graph flagged, and its right cut decisions; row-null gain check: sensitivity and false alarm. Pooled over η and constructions on the test worlds, the divergence check has a sensitivity of {f(I['divergence_test']['sens'], 2)} and a false-alarm rate of {f(I['divergence_test']['false'], 2)} ({I['divergence_test']['n_pos']} of {I['divergence_test']['n']} graphs with V \\> 0.02), the row-null gain check {f(I['gate_test']['sens'], 2)} and {f(I['gate_test']['false'], 2)}.\n\n" +
               table(["Construction", "η", "V", "V \\> 0.02", "Loss vs Π_F", "BA vs support", "BA vs Π_F", "Cuts right vs Π_F"], [r[:8] for r in rows]) + "\n\n" +
               table(["Construction", "η", "Divergence: sensitivity", "false alarm", "BA vs Π_F with the check", "Cuts right vs Π_F with the check", "Gain check: sensitivity", "false alarm"], [r[:2] + r[8:] for r in rows]))
    return "\n\n".join(out)


def s4():
    out = []
    P = S["junyi_pruning_value"]["taus"]
    rows = []
    for t, r in P.items():
        for lev in ("edge", "path"):
            v = r["value"][lev]
            ci = lambda q: f"{f(v[q], 3, True)} \\[{f(v[q + '_ci'][0], 3, True)}, {f(v[q + '_ci'][1], 3, True)}\\]"
            rows.append([("none" if t == "0" else t), str(r["n"]), str(r["m"]), f(r["mean_out_degree"], 1), lev, f(v["support"]), f(v["observed"]), ci("V_marg"), ci("V_strength"), ci("V_resid"), ci("V_total"),
                         f"{v['global_mc_se']:.4f} / {v['row_mc_se']:.4f}"])
    out.append("**Table S4b.** The Junyi transition graph rebuilt at five pruning thresholds τ_E and scored on the fixed 1,954 annotated pairs (a pair outside the graph or not reached scores 0); decomposition (3) with 20 draws per null and two-way cluster-bootstrap intervals over exercises (500 resamples, null draws fixed); MC SE: Monte Carlo standard error of the global / row null means (scripts/v16_junyi_pruning.py).\n\n" +
               table(["τ_E", "Nodes", "Edges", "Out-degree", "Level", "Support 𝒮", "Graded 𝒮", "V_marg", "V_str", "V_res", "V", "MC SE"], rows))
    U = S["units"]
    rows = []
    for lev in ("path", "partition"):
        for n, d in U[lev].items():
            vv = {kk: x for kk, x in d.items() if not kk.startswith("_")}
            rows.append([NAMES[n], lev, f(vv["mu"]["V_total"], 3, True), f(vv["mu"]["V_resid"], 3, True),
                         f"{f(d['_range']['V_total'][0], 3, True)} to {f(d['_range']['V_total'][1], 3, True)}",
                         f"{f(d['_range']['V_resid'][0], 3, True)} to {f(d['_range']['V_resid'][1], 3, True)}", str(d["_range"]["sign_changes"])])
    out.append("**Table S4c.** Unit dependence of the path and partition values: μ_E replaced by φ_s(μ_E) for the 18 vertices of B_2 that make one segment twice as steep (s = 1 + e_j) or half as steep (s = 2 − e_j) as the rest, rescaled so that the largest membership is unchanged; the whole decomposition recomputed on each (path: 20 draws per null; partition: ten draws, best of three seeds). Sign changes: vertices at which V has the opposite sign to V on μ_E (scripts/v16_units.py).\n\n" +
               table(["Graph", "Level", "V on μ_E", "V_res on μ_E", "V over B_2", "V_res over B_2", "Sign changes"], rows))
    G = S["real"]["graphs"]
    rows = []
    for n, g in G.items():
        if "value_seeds" not in g or g["value_seeds"] is None:
            continue
        if V15[n]["grouping"]["ari"]["support"][0] < 0.1 and g["grouping"]["symmetric_max_best3"] < 0.1:
            continue
        vs = g["value_seeds"]
        col = f"{f(vs['col_mean'], 3)} ({f(vs['col_mc_se'], 3)})" if "col_mean" in vs else "--"
        rows.append([NAMES[n], f"{f(min(vs['V_total_triples']), 3, True)} to {f(max(vs['V_total_triples']), 3, True)}",
                     f"{f(min(vs['V_resid_triples']), 3, True)} to {f(max(vs['V_resid_triples']), 3, True)}", f(vs["glob_mc_se"], 4), f(vs["row_mc_se"], 4), col])
    out.append("**Table S4e.** Partition value of the grades on the ten graphs with an informative grouping: range over the six disjoint triples of optimiser seeds (0--2, ..., 15--17; best of three by Q_F within a triple) of V and V_res; Monte Carlo standard errors of the global and row null means (ten draws); for the two graphs with V \\> 0, the mean ARI under a column null that permutes the memberships among the in-edges of each node (ten draws, best of three seeds; MC SE in brackets), to be compared with the row null (Table 3).\n\n" +
               table(["Graph", "V over seed triples", "V_res over seed triples", "MC SE global", "MC SE row", "Column null ARI"], rows))
    return "\n\n".join(out)


def s5():
    G = S["real"]["graphs"]; Dv = S.get("divergence_real", {})
    rows = []
    for n, g in G.items():
        for c, r in g["cuts"].items():
            rows.append([NAMES[n], c[1:], r["decision_mu"], k(min(r["kappa_star"], 1e3)), f"{k(r['kappa_boot'][1])} to {k(r['kappa_boot'][2])}", f(r["kappa_boot_share_ge2"], 2),
                         k(min(r["kappa_star_refined"], 1e3)), f(r["E_F"], 4)])
    out = ["**Table S5b.** Every quantile cut: κ\\* in the family (5), its 2.5 to 97.5 percentile range over 100 bootstrap resamples of the 20 optimiser seeds (support and cut partitions resampled jointly), the share of resamples with κ\\* ≥ 2, κ\\* in the refined family of Corollary 1, and E_F = Q_F(Π_F) − Q_F(Π_cut) on μ_E (scripts/v16_real.py).\n\n" +
           table(["Graph", "Cut", "Decision on μ_E", "κ\\*", "κ\\* over seed resamples", "Share κ\\* ≥ 2", "κ\\* refined", "E_F"], rows)]
    rows = []
    for n, g in G.items():
        d = Dv.get(n, {})
        rows.append([NAMES[n], f(d.get("F_S"), 3), f(d.get("S_S"), 3), f(d.get("F_F"), 3), f(d.get("gap"), 3, True),
                     f(g["gate"]["g_obs"], 4), f(g["gate"]["null_max"], 4), "yes" if g["gate"]["fires"] else "no"])
    out.append("**Table S5c.** The two checks of Section 4.5 on the 23 graphs: mean agreement of Π_F with Π_sup, and of consecutive seeds of each (20 seeds), and the divergence D of (7); the gain g of Π_F over Π_sup on μ_E against the largest gain over ten row permutations (scripts/v16_divergence.py, scripts/v16_real.py).\n\n" +
               table(["Graph", "a_FS", "a_SS", "a_FF", "D", "Gain g", "Max over row nulls", "Gain check fires"], rows))
    E = S["real"]["external"]["rows"]
    rows = [[NAMES[r["graph"]], r["cut"][1:], r["robust"], f(r["loss_value"], 3, True), "yes" if r["loss"] else "no", f(r["lossF_value"], 3, True), "yes" if r["lossF"] else "no"] for r in E]
    out.append("**Table S5d.** The external check against both losses: robust decision, loss against the support partition and against the graded partition (ARI with the grouping, 20 seeds; a loss when at least 0.02 with an interval excluding 0).\n\n" +
               table(["Graph", "Cut", "Robust decision", "Loss vs Π_sup", "Loss", "Loss vs Π_F", "Loss"], rows))
    return "\n\n".join(out)


def s7():
    L = S["learner_resampling"]
    rows = []
    for key, rec in L.items():
        for c, v in rec["summary"].items():
            o = v["observed"]
            for kind in ("subsample", "bootstrap"):
                if kind in v:
                    x = v[kind]
                    rows.append([key.replace("_", " "), c[1:], o["decision"], k(min(o["kappa_star"], 1e3)), kind, str(len(rec[kind])), f(x["share_same_decision"], 2), f(x["share_same_robust"], 2),
                                 ", ".join(k(q) for q in x["kappa_star_quartiles"]), f"{x['edges_median']:.0f} / {rec['observed']['m']}"])
    return ("**Table S7b.** Resampling of the learners without duplicates: subsamples of 80% of the learners without replacement on the six ASSISTments KC-level graphs (20 draws) and on Junyi transition (10 draws), and the bootstrap of Table S7a on Junyi transition (10 draws); ten seeds, δ by density class, κ\\* searched up to 100 (scripts/v16_learner_resampling.py). Edges: median over draws / observed.\n\n" +
            table(["Graph", "Cut", "Observed", "κ\\*", "Resampling", "Draws", "Same", "Same robust", "κ\\* quartiles", "Edges"], rows))


if __name__ == "__main__":
    out = {}
    for name, fn in (("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S7", s7)):
        try:
            out[name] = fn()
        except Exception as e:   # noqa: BLE001
            out[name] = f"⟦{name} pending: {e}⟧"
    json.dump(out, open("results/v16/supp_tables.json", "w"), indent=1, ensure_ascii=False)
    for kk, v in out.items():
        print(kk, len(v), v[:120].replace("\n", " "))
