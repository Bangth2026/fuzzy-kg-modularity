"""Leiden on Q_F on every real graph at four resolutions (Tables 2-4), with one seed
protocol for every table of the paper: seeds 0-4, optimisation until convergence,
the partition with the highest Q_F kept.  The well-connectedness (Claim 1) and
monotonicity (Lemma 1(iii)) checks are run separately with ten recorded iterations.

Also: graph statistics, and the agreement of the gamma = 1 partition with the
reference partitions available for each corpus (weakly connected components; the
skill and the modal assignment of each item on the ASSISTments item-level graphs; the
topic and area of each exercise on Junyi), with a description of the largest
communities of the 2017 item-level graph by their dominant skills.

Usage: python scripts/real_graph_tables.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzykg import FuzzyGraph, fuzzy_leiden  # noqa: E402
import scipy.sparse.csgraph as csg  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami  # noqa: E402

NAMES = ["mooccubex_cs", "mooccubex_cs_full", "assist2009_kc", "assist2009_item", "assist2015_kc", "assist2017_kc",
         "assist2017_item", "junyi_copractice", "assist2009_kc_transition", "assist2015_kc_transition",
         "assist2017_kc_transition", "assist2017_item_transition", "junyi_transition"]
GAMMAS = [0.5, 1.0, 2.0, 5.0]
SEEDS = range(5)


def load(name):
    df = pd.read_csv(f"data/graphs/{name}_edges.csv"); nd = pd.read_csv(f"data/graphs/{name}_nodes.csv")
    return FuzzyGraph.from_edges(zip(df.src, df.dst, df.mu), n=len(nd)), nd.label.astype(str).tolist()


def best(G, g):
    return max((fuzzy_leiden(G, g, seed=s, n_iterations=0) for s in SEEDS), key=lambda r: r.Q_F)


def references(name, labels):
    refs = {}
    if name.startswith("assist2009_item"):
        df = pd.read_csv("data/raw/assist2009/skill_builder_data.csv", usecols=["problem_id", "skill_id"], encoding="latin-1").dropna()
        first = df.drop_duplicates("problem_id").set_index("problem_id")["skill_id"]
        first.index = first.index.astype(str)
        refs["skill"] = [str(first.get(l.split(".")[0], "NA")) for l in labels]
    if name.startswith("assist2017_item"):
        df = pd.read_csv("data/raw/assist2017/anonymized_full_release_competition_dataset.csv", usecols=["problemId", "skill", "assignmentId"], low_memory=False).dropna()
        df["problemId"] = df["problemId"].astype(str)
        sk = df.groupby("problemId")["skill"].agg(lambda x: x.value_counts().index[0])
        asg = df.groupby("problemId")["assignmentId"].agg(lambda x: x.value_counts().index[0])
        refs["skill"] = [str(sk.get(l.split(".")[0], "NA")) for l in labels]
        refs["assignment"] = [str(asg.get(l.split(".")[0], "NA")) for l in labels]
    if name.startswith("junyi"):
        ex = pd.read_csv("data/raw/junyi/junyi/junyi_Exercise_table.csv", encoding="utf-8-sig")
        t = dict(zip(ex.name, ex.topic)); a = dict(zip(ex.name, ex.area))
        refs["topic"] = [str(t.get(l, "NA")) for l in labels]
        refs["area"] = [str(a.get(l, "NA")) for l in labels]
    return refs


def main():
    out = {}
    for name in NAMES:
        G, labels = load(name)
        ncomp, comp = csg.connected_components(G.symmetrised(), directed=False)
        row = {"n": G.n, "m": G.m, "density": G.m / (G.n * (G.n - 1)), "reciprocity": G.reciprocity(),
               "mu_min": float(G.mu.min()), "mu_mean": float(G.mu.mean()), "mu_sd": float(G.mu.std()), "mu_max": float(G.mu.max()),
               "components": int(ncomp), "largest_component": int(np.bincount(comp).max()),
               "acyclic": bool(G.to_igraph().is_dag()), "gammas": {}}
        for g in GAMMAS:
            b = best(G, g)
            chk = [fuzzy_leiden(G, g, seed=s, n_iterations=10) for s in SEEDS]
            row["gammas"][str(g)] = {"Q_F": b.Q_F, "k": b.n_communities, "largest": int(b.sizes[0]), "smallest": int(b.sizes[-1]),
                                     "conn_violations": int(sum(r.connected_violations for r in chk) + b.connected_violations),
                                     "mono_violations": int(sum(r.monotone_violations for r in chk))}
            if g == 1.0:
                part = b.membership
        row["ARI_vs_components"] = float(ari(comp, part))
        for rname, ref in references(name, labels).items():
            row[f"ARI_vs_{rname}"] = float(ari(ref, part)); row[f"AMI_vs_{rname}"] = float(ami(ref, part))
            row[f"n_{rname}"] = int(len(set(ref)))
            if rname == "skill" and name == "assist2017_item":
                desc = []
                for c in np.argsort(-np.bincount(part))[:4]:
                    members = [ref[i] for i in np.flatnonzero(part == c)]
                    vc = pd.Series(members).value_counts()
                    desc.append({"size": len(members), "top_skills": vc.head(3).to_dict(), "purity": float(vc.iloc[0] / len(members)),
                                 "n_skills": int(vc.size)})
                row["communities_by_skill"] = desc
                ncs = pd.Series(ref).nunique()
                row["skill_purity_weighted"] = float(sum(pd.Series([ref[i] for i in np.flatnonzero(part == c)]).value_counts().iloc[0] for c in np.unique(part)) / G.n)
                row["skill_purity_components"] = float(sum(pd.Series([ref[i] for i in np.flatnonzero(comp == c)]).value_counts().iloc[0] for c in np.unique(comp)) / G.n)
        out[name] = row
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "gammas"}),
              {g: (round(v["Q_F"], 3), v["k"], v["smallest"]) for g, v in row["gammas"].items()}, flush=True)
        json.dump(out, open("results/real_graph_tables.json", "w"), indent=1)


if __name__ == "__main__":
    main()
