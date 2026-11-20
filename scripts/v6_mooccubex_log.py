#!/usr/bin/env python3
"""Version 4.3.0. How much of the MOOCCubeX exercise log falls on Computer-Science concepts.

Input: the records of relations/user-problem.json (22.45 GB, one full streaming pass) whose
problem is linked, in relations/concept-problem.txt, to a concept of the CS discipline
(计算机科学与技术); the pass itself is scripts/v6_mooccubex_scan.py.  The records are mapped to
the concepts of the expert graph (287 concepts) and to all CS concepts, ordered by submit time
within a learner, and the counts that the two constructions of Section 4 need are formed:
moves between consecutive concept runs (transition) and ordered co-occurrences within a
horizon of 10 interactions (co-practice); a pair is usable at support 20.
Writes results/v6/mooccubex_log.json.
Usage: python scripts/v6_mooccubex_log.py cs_records.jsonl concept-problem.txt scan_summary.json
"""
import json, sys
from collections import Counter, defaultdict
import pandas as pd

rec_path, cp_path, summ_path = sys.argv[1:4]
nodes = set(pd.read_csv("data/graphs/mooccubex_cs_full_nodes.csv").label.astype(str))
p2c = defaultdict(set)
for line in open(cp_path, encoding="utf-8"):
    cid, pid = line.rstrip("\n").split("\t")[:2]
    if cid.startswith("K_"):
        name, field = cid[2:].rsplit("_", 1)
        if field == "计算机科学与技术":
            p2c[pid].add(name)
rows = []
for line in open(rec_path, encoding="utf-8"):
    r = json.loads(line)
    rows.append((r["user_id"], r["submit_time"] or "", r["problem_id"]))
df = pd.DataFrame(rows, columns=["user", "time", "problem"]).drop_duplicates()
out = {"scan": json.load(open(summ_path)), "records_cs": int(len(df)), "learners": int(df.user.nunique()),
       "problems": int(df.problem.nunique())}
for scope in ("expert_graph", "all_cs"):
    seqs = defaultdict(list)
    for u, t, p in df.sort_values(["user", "time"]).itertuples(index=False):
        cs = sorted(c for c in p2c[p] if scope == "all_cs" or c in nodes)
        for c in cs:
            seqs[u].append(c)
    conc = Counter(c for s in seqs.values() for c in s)
    T = Counter(); N = Counter()
    for s in seqs.values():
        runs = [c for i, c in enumerate(s) if i == 0 or c != s[i - 1]]
        for a, b in zip(runs, runs[1:]):
            T[(a, b)] += 1
        for i, a in enumerate(s):
            for b in s[i + 1:i + 11]:
                if a != b:
                    N[(a, b)] += 1
    def usable(C):
        seen = set(); k = 0
        for (a, b), v in C.items():
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            if v + C.get((b, a), 0) >= 20:
                k += 1
        return k, len(seen)
    tu, tp = usable(T); nu, npairs = usable(N)
    out[scope] = {"records": int(sum(conc.values())), "learners": int(sum(1 for s in seqs.values() if s)),
                  "concepts_seen": len(conc), "concepts_with_20": int(sum(v >= 20 for v in conc.values())),
                  "max_records_per_concept": int(max(conc.values())) if conc else 0,
                  "transition_pairs": tp, "transition_pairs_support20": tu,
                  "copractice_pairs": npairs, "copractice_pairs_support20": nu}
json.dump(out, open("results/v6/mooccubex_log.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=1))
