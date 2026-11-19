#!/usr/bin/env python3
"""Version 4.0.0. Collects the v4 results into the tables of the revised manuscript
(results/v4/tables.md) and a flat dictionary of every number quoted in the text
(results/v4/paper_numbers.json)."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
R = "results/v4"
NUM = {}


def J(f):
    p = os.path.join(R, f)
    return json.load(open(p)) if os.path.exists(p) else {}


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def zf(z):
    return ">10" if z is not None and z > 10 else (f"{z:.1f}" if z is not None else "--")


NAMES = {"assist2012": "ASSISTments 2012", "algebra2005": "Algebra 2005", "bridge2006": "Bridge to Algebra 2006",
         "eedi": "Eedi", "ednet": "EdNet"}


def table1_new(out):
    st = J("new_corpora_stats.json"); a1 = J("algorithm1_new.json")
    L = ["| Graph | Construction | Learners | Interactions | Nodes | Edges | Density | Reciprocity | μ_E mean (SD) | Components | Q_F | Communities | ARI with grouping |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in NAMES:
        s = st.get(c)
        if not s:
            continue
        grp = json.load(open(f"data/graphs/{c}_groups.json")) if os.path.exists(f"data/graphs/{c}_groups.json") else None
        for k in ("copractice", "transition"):
            name = f"{c}_{k}"
            e = pd.read_csv(f"data/graphs/{name}_edges.csv"); lab = pd.read_csv(f"data/graphs/{name}_nodes.csv").label.astype(str).tolist()
            n = len(lab); m = len(e)
            act = np.unique(np.r_[e.src, e.dst]); na = act.size
            A = csr_matrix((np.ones(m), (e.src, e.dst)), shape=(n, n))
            ncomp = connected_components(A[act][:, act], directed=True, connection="weak")[0]
            rec = float(((A.multiply(A.T)) > 0).sum() / m)
            r = a1.get(name, {})
            ag = "--"
            if grp is not None and "membership" in r:
                mem = np.asarray(r["membership"]); g = np.array([grp.get(l) for l in lab], dtype=object)
                has = np.array([x is not None for x in g]) & np.isin(np.arange(n), act)
                ag = fmt(ari(pd.factorize(pd.Series(g[has]).astype(str))[0], mem[has]), 2)
                NUM[f"{name}.ari_grouping"] = float(ag)
            L.append(f"| {NAMES[c]} | {k.replace('copractice','co-practice')} | {s['learners']:,} | {s['interactions']:,} | {na} | {m:,} | "
                     f"{m / (na * (na - 1)):.3f} | {rec:.2f} | {e.mu.mean():.2f} ({e.mu.std():.2f}) | {ncomp} | {fmt(r.get('Q_F'))} | {r.get('k', '--')} | {ag} |")
            NUM[f"{name}.n"] = na; NUM[f"{name}.m"] = m; NUM[f"{name}.density"] = m / (na * (na - 1)); NUM[f"{name}.reciprocity"] = rec
    out.append("### Table 1 (new rows)\n\n" + "\n".join(L) + "\n")


def table3(out):
    a1 = J("algorithm1_new.json"); dc = J("dcfloor.json")
    old = json.load(open("results/calibrated_null_test.json"))
    L = ["| Graph | Q_F | k | Degree null ΔQ (z) | Floor ΔQ* (p_in) | Outcome | Structure-matched floor (p_in) | Outcome, matched floor | Weights global ΔQ (z) | Weights row ΔQ (z) | r* global / row |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in list(old.items()) + list(a1.items()):
        f = dc.get(name, {})
        def rstar(p):
            return fmt(p.get("detectable_r_pb"), 2) if p else "--"
        L.append(f"| {name} | {fmt(d['Q_F'])} | {d['k']} | {d['degree']['delta_Q']:+.3f} ({zf(d['degree']['z'])}) | {fmt(d['floor']['floor'])} ({fmt(d['floor']['floor_p_in'],2)}) | "
                 f"{f.get('outcome_density','--')} | {fmt(f.get('floor_dc'))} ({fmt(f.get('floor_dc_p_in'),2)}) | {f.get('outcome_dc','--')} | "
                 f"{d['weights_global']['delta_Q']:+.4f} ({zf(d['weights_global']['z'])}) | {d['weights_row']['delta_Q']:+.4f} ({zf(d['weights_row']['z'])}) | "
                 f"{rstar(d.get('power_global'))} / {rstar(d.get('power_row'))} |")
    out.append("### Table 3 (all graphs, with the structure-matched floor)\n\n" + "\n".join(L) + "\n")
    ch = [(k, v["outcome_density"], v["outcome_dc"]) for k, v in dc.items() if v["outcome_density"] != v["outcome_dc"]]
    NUM["dc.changed"] = ch; NUM["dc.n"] = len(dc)
    out.append(f"Outcome changes under the matched floor: {ch}\n")


def table_sens(out):
    s = J("sensitivity.json")
    L = ["| Log / construction | Variants | Density (range) | Reciprocity (range) | Weights row z (range) | Weights global z (range) | Q_F drop q0.50 (range) | Q_F drop q0.75 (range) |",
         "|---|---|---|---|---|---|---|---|"]
    keys = sorted({"/".join(k.split("/")[:2]) for k in s})
    for g in keys:
        rows = [v for k, v in s.items() if k.startswith(g + "/")]
        def rg(f, d=3):
            v = [f(r) for r in rows]; return f"{min(v):.{d}f} to {max(v):.{d}f}"
        L.append(f"| {g} | {len(rows)} | {rg(lambda r: r['density'])} | {rg(lambda r: r['reciprocity'], 2)} | {rg(lambda r: r['weights_row']['z'], 1)} | "
                 f"{rg(lambda r: r['weights_global']['z'], 1)} | {rg(lambda r: r['qf_drop']['q0.50'])} | {rg(lambda r: r['qf_drop']['q0.75'])} |")
    out.append("### Sensitivity\n\n" + "\n".join(L) + "\n")
    j = {k: v for k, v in s.items() if k.startswith("junyi/") and "expert_rho" in v}
    if j:
        L = ["| Junyi variant | Edges | direct | hop | max-min | product |", "|---|---|---|---|---|---|"]
        for k, v in j.items():
            e = v["expert_rho"]
            L.append(f"| {k} | {v['m']} | {e['direct_edge']:.3f} | {e['hop_baseline']:.3f} | {e['min']:.3f} | {e['product']:.3f} |")
        out.append("### Junyi expert criterion under construction variants\n\n" + "\n".join(L) + "\n")


def table_nc(out):
    import glob
    L = ["| Log | Construction | Events | Direct | Hop | Selected (op) | Popularity | Recency | Bigram | Markov-2 | GRU | GRU − selected [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{R}/nextconcept_*.json")):
        r = json.load(open(f)); s = r["scores"]; p = r["paired"].get("gru-selected_operator", {})
        tag = os.path.basename(f)[12:-5]
        L.append(f"| {r['release']} {r['level']}{' (unsuggested)' if 'unsuggested' in tag else ''} | {r['construction']} | {r['n_test_events']:,} | {s['direct_edge']['mrr_learner_mean']:.3f} | "
                 f"{s['hop_distance']['mrr_learner_mean']:.3f} | {s['selected_operator']['mrr_learner_mean']:.3f} ({r['selected_operator']}) | {s['popularity']['mrr_learner_mean']:.3f} | "
                 f"{s['recency']['mrr_learner_mean']:.3f} | {s['bigram']['mrr_learner_mean']:.3f} | {s['markov2_backoff']['mrr_learner_mean']:.3f} | {s['gru']['mrr_learner_mean']:.3f} | "
                 f"{p.get('diff', float('nan')):+.3f} [{p.get('ci', [np.nan, np.nan])[0]:+.3f}, {p.get('ci', [np.nan, np.nan])[1]:+.3f}] |")
        NUM[f"nc.{tag}"] = {k: v["mrr_learner_mean"] for k, v in s.items()}
    out.append("### Next concept with sequence baselines\n\n" + "\n".join(L) + "\n")


def table_protocol(out):
    d = J("protocol_external.json")
    L = ["| Tilt | Mean r_pb | Cut cells | ρ(audit excess, ARI loss) | Flagged (Q_F drop) | Resolved ARI losses | Precision | Sensitivity | Precision, rank-coded | Cuts that raise ARI |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tilt in ("0", "0.5", "1", "2", "4"):
        X = []; Y = []; F = []; FR = []; Rr = []; rp = []
        for k, v in d.items():
            g, t = k.split("/tilt")
            if t != tilt:
                continue
            rp.append(v["info"]["r_pb_internal"] or 0)
            for c, r in v["rows"].items():
                if c in ("Pi_F", "support") or r.get("empty"):
                    continue
                X.append(r["qf_drop"] - v["rows"]["support"]["qf_drop"]); Y.append(r["ari_loss_vs_support"])
                F.append(r["flag_qf_drop"]); FR.append(r["flag_rank_drop"]); Rr.append(r["ari_loss_resolved"])
        X, Y, F, FR, Rr = map(np.array, (X, Y, F, FR, Rr))
        if not len(X):
            continue
        rho = spearmanr(X, Y).statistic; prec = (F & Rr).sum() / max(1, F.sum()); sens = (F & Rr).sum() / max(1, Rr.sum())
        precr = (FR & Rr).sum() / max(1, FR.sum()); raise_ = int((Y < -1e-9).sum())
        NUM[f"protocol.tilt{tilt}"] = {"rho": float(rho), "flagged": int(F.sum()), "losses": int(Rr.sum()), "precision": float(prec),
                                       "sensitivity": float(sens), "precision_rank": float(precr), "cells": int(len(X)), "raise": raise_,
                                       "flagged_rank": int(FR.sum()), "npv": float(((~F) & (~Rr)).sum() / max(1, (~F).sum())), "mean_rpb": float(np.mean(rp))}
        L.append(f"| {tilt} | {np.mean(rp):.2f} | {len(X)} | {rho:.2f} | {F.sum()} | {Rr.sum()} | {prec:.2f} | {sens:.2f} | {precr:.2f} | {raise_} |")
    out.append("### Protocol against external groupings\n\n" + "\n".join(L) + "\n")


def table_scaling(out):
    s = J("scaling.json")
    if not s:
        return
    L = ["| n | mean out-degree | edges | Leiden on Q_F (s) | multiplex (s) | degree draw (s) | weights draw (s) | matched planted graph (s) | product paths, all sources (s) | max-min paths (s) | peak extra memory, Leiden / degree draw (MB) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in (8, 32):
        for n in (1000, 2000, 5000, 10000, 20000):
            g = lambda op, f="time_s": s.get(f"n{n}_d{d}_{op}", {}).get(f)
            m = g("leiden", "m")
            if m is None:
                continue
            L.append(f"| {n:,} | {d} | {m:,} | {fmt(g('leiden'),2)} | {fmt(g('multiplex'),2)} | {fmt(g('degree_draw'),2)} | {fmt(g('weights_draw'),2)} | "
                     f"{fmt(g('dc_planted'),2)} | {fmt(g('paths_product','time_all_sources_s'),1)} | {fmt(g('paths_maxmin','time_all_sources_s'),1)} | "
                     f"{fmt(g('leiden','peak_extra_MB'),0)} / {fmt(g('degree_draw','peak_extra_MB'),0)} |")
    out.append("### Scaling\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    out = []
    for f in (table1_new, table3, table_sens, table_nc, table_protocol, table_scaling):
        try:
            f(out)
        except Exception as e:  # report and continue
            out.append(f"{f.__name__}: {e!r}\n")
    for k in ("index_negativity_seeds.json", "junyi_labels.json"):
        NUM[k] = J(k)
    open(f"{R}/tables.md", "w").write("\n".join(out))
    json.dump(NUM, open(f"{R}/paper_numbers.json", "w"), indent=1, default=str)
    print("\n".join(out))
