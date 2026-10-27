"""Tables 2-4 and the reference-partition and null-model comparisons on the real
graphs, in one run (Section 8.1).  Usage: python scripts/real_graph_tables.py
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden
from fuzzykg.leiden import best_of_seeds
from fuzzykg.resolution import null_model_zscore
import scipy.sparse.csgraph as csg
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami, normalized_mutual_info_score as nmi

NAMES = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc", "assist2017_item"]

def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu))

def table_2_4():
    out = {}
    for name in NAMES:
        G = load(name)
        gammas, seeds = ([0.5, 1.0], 10) if name.startswith("mooccubex") else ([0.5, 1.0, 2.0, 5.0], 5)
        rows = {}
        for g in gammas:
            runs = [fuzzy_leiden(G, g, seed=s, n_iterations=10) for s in range(seeds)]
            best = max(runs, key=lambda r: r.Q_F)
            rows[str(g)] = {"Q_F": best.Q_F, "k": best.n_communities, "sizes": best.sizes[:6],
                            "conn_violations": sum(r.connected_violations for r in runs),
                            "mono_violations": sum(r.monotone_violations for r in runs)}
        out[name] = rows
        print(name, {g: (round(v["Q_F"], 3), v["k"]) for g, v in rows.items()}, flush=True)
    return out

def reference_partitions():
    """Problem-set (skill) reference partition for the item-level graphs, and
    components for all; NMI/ARI/AMI of the Fuzzy-Leiden partition at gamma = 1."""
    out = {}
    raw = {"assist2009_item": ("data/raw/assist2009/skill_builder_data.csv", "problem_id", "skill_id", "latin-1"),
           "assist2017_item": ("data/raw/assist2017/anonymized_full_release_competition_dataset.csv", "problemId", "skill", None)}
    for name in NAMES:
        G = load(name)
        nodes = pd.read_csv(f"data/graphs/{name}_nodes.csv")
        ncomp, comp = csg.connected_components(G.symmetrised(), directed=False)
        part = best_of_seeds(G, 1.0, seeds=range(5), n_iterations=0)
        row = {"n": G.n, "m": G.m, "components": int(ncomp), "largest_component": int(np.bincount(comp).max()),
               "k_leiden_gamma1": part.n_communities,
               "ARI_vs_components": ari(comp, part.membership), "AMI_vs_components": ami(comp, part.membership)}
        if name in raw and os.path.exists(raw[name][0]):
            path, pid, sid, enc = raw[name]
            df = pd.read_csv(path, usecols=[pid, sid], encoding=enc, low_memory=False).dropna()
            first = df.drop_duplicates(subset=[pid]).set_index(pid)[sid].astype(str)
            labels = nodes["label"].astype(str)
            ref = labels.map(lambda x: first.get(x, first.get(int(float(x)) if x.replace('.', '').isdigit() else x, "NA")))
            ref = ref.fillna("NA").astype(str).values
            row["n_problem_sets"] = int(len(set(ref)))
            row["ARI_vs_problem_set"] = ari(ref, part.membership); row["AMI_vs_problem_set"] = ami(ref, part.membership)
            row["NMI_vs_problem_set"] = nmi(ref, part.membership)
            row["ARI_components_vs_problem_set"] = ari(ref, comp)
        out[name] = row; print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out

def nulls():
    """Two null models at gamma = 1 with the same five seeds for observed and null:
    degree-preserving rewiring + membership permutation, and membership
    permutation on the fixed topology."""
    out = {}
    for name in NAMES:
        G = load(name)
        a = null_model_zscore(G, 1.0, n_perm=20, seeds=(0, 1, 2, 3, 4), rewire=True)
        b = null_model_zscore(G, 1.0, n_perm=20, seeds=(0, 1, 2, 3, 4), rewire=False)
        out[name] = {"rewire_and_permute": a, "permute_weights_only": b}
        print(name, "Q=%.3f | rewire: null %.3f±%.4f dQ %.3f z %.1f | weights-only: null %.3f±%.4f dQ %.3f z %.1f" % (
            a["Q_F_obs"], a["null_mean"], a["null_sd"], a["Q_F_obs"] - a["null_mean"], min(a["z"], 9999),
            b["null_mean"], b["null_sd"], b["Q_F_obs"] - b["null_mean"], min(b["z"], 9999)), flush=True)
    return out

if __name__ == "__main__":
    res = {"tables_2_4": table_2_4(), "reference_partitions": reference_partitions(), "nulls": nulls()}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/real_graph_tables.json", "w"), indent=1)
