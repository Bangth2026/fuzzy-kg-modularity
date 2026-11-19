#!/usr/bin/env python3
"""Version 4.0.0. Descriptive statistics of every log after the stated preprocessing (learners,
interactions, nodes with at least one interaction, correct rate, interactions per learner),
for the reply to the request for a descriptive table. Writes results/v4/corpus_stats.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_any  # noqa: E402
from fuzzykg.construction import ConstructionConfig, preprocess  # noqa: E402
out = "results/v4/corpus_stats.json"; res = json.load(open(out)) if os.path.exists(out) else {}
for rel, lev in [("2009", "kc"), ("2009", "item"), ("2015", "kc"), ("2017", "kc"), ("2017", "item"), ("junyi", "kc"),
                 ("assist2012", "kc"), ("algebra2005", "kc"), ("bridge2006", "kc"), ("eedi", "kc"), ("ednet", "kc")]:
    k = f"{rel}_{lev}"
    if k in res:
        continue
    d = preprocess(load_any(rel, lev), ConstructionConfig())
    n = d.groupby("user_id").size()
    res[k] = {"learners": int(d.user_id.nunique()), "interactions": int(len(d)), "nodes": int(d.kc.nunique()),
              "correct_rate": float(d.correct.mean()), "per_learner_median": float(n.median()), "per_learner_mean": float(n.mean())}
    json.dump(res, open(out, "w"), indent=1); print(k, res[k], flush=True)
