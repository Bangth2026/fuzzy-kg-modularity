#!/usr/bin/env python3
"""Version 4.0.0. Held-out next-concept prediction with sequence baselines (reply to the
request for stronger baselines than the hop count).

Same protocol as scripts/heldout_transitions.py (70/10/20 learner split by hash, graph built on
training learners, candidates = every node of the training graph except the source, expected
reciprocal rank under random tie-breaking, learner mean, test learners capped at 2,000). Scores:

  graph    direct edge, hop distance on the support, product best path, and the composition
           operator selected on validation learners among the ten of heldout_transitions.py
  sequence popularity (training frequency), recency (how recently this learner last visited the
           candidate), bigram N(a -> .), second-order Markov N(p, a -> .) backed off to the bigram,
           and a GRU next-run model trained on the training learners.

Usage: python scripts/v4_nextconcept_baselines.py --release 2009 --level kc --construction transition
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from heldout_transitions import (OPERATORS, split_id, strength_matrix, paired, holm)  # noqa: E402
from data.loaders import load_any  # noqa: E402
from fuzzykg.construction import (ConstructionConfig, TransitionConfig, build_fuzzy_graph,  # noqa: E402
                                  build_transition_graph, preprocess)


def runs_by_learner(d, uids):
    """Run-collapsed KC sequences of the given learners, in log order, as arrays of category codes
    (d["kc"] must be categorical; KC labels are recovered from d["kc"].cat.categories)."""
    sub = d[d.user_id.isin(set(uids))]
    u = sub["user_id"].to_numpy(); k = sub["kc"].cat.codes.to_numpy().astype(np.int32)
    out = {}
    if len(u) == 0:
        return out
    cut = np.flatnonzero(u[1:] != u[:-1]) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(u)]):
        s = k[a:b]
        keep = np.ones(s.size, dtype=bool); keep[1:] = s[1:] != s[:-1]
        out[u[a]] = s[keep]
    return out


def events(seqs, idx):
    """Test events: (learner, position i) with run i and run i+1 both graph nodes; idx maps a
    category code to a node index (-1 when the KC is not a node)."""
    ev = []
    for u, s in seqs.items():
        ii = idx[s]
        ok = np.flatnonzero((ii[:-1] >= 0) & (ii[1:] >= 0))
        ev.extend((u, int(i)) for i in ok)
    return ev


def expected_rr(S, src, tgt):
    """Expected reciprocal rank (random ties), reciprocal of the average tie rank; S rows = events."""
    S = S.astype(float).copy()
    S[np.arange(S.shape[0]), src] = -np.inf
    n = S.shape[1] - 1
    harm = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 2))])
    v = S[np.arange(S.shape[0]), tgt][:, None]
    above = (S > v).sum(1); ties = (S == v).sum(1)
    rr = (harm[above + ties] - harm[above]) / ties
    rra = 1.0 / (above + (ties + 1) / 2.0)
    return rr, rra


def summarise(rr, rra, uid, B=1000, seed=0):
    df = pd.DataFrame({"u": uid, "rr": rr, "rra": rra})
    per = df.groupby("u", sort=True)["rr"].mean()
    m = per.to_numpy(); rng = np.random.default_rng(seed)
    boots = m[rng.integers(0, m.size, (B, m.size))].mean(axis=1)
    return {"mrr_learner_mean": float(m.mean()), "mrr_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mrr_learner_mean_average_rank_ties": float(df.groupby("u", sort=True)["rra"].mean().mean()),
            "n_events": int(rr.size)}, per


class GRUNext:
    def __init__(self, vocab, n_out, dim=64, hid=128, seed=0):
        import torch
        torch.manual_seed(seed)
        self.t = torch
        self.emb = torch.nn.Embedding(vocab + 1, dim, padding_idx=0)
        self.rnn = torch.nn.GRU(dim, hid, batch_first=True)
        self.out = torch.nn.Linear(hid, n_out)
        self.params = list(self.emb.parameters()) + list(self.rnn.parameters()) + list(self.out.parameters())

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h)


def gru_scores(train_seqs, val_seqs, test_seqs, vocab_index, idx, n, epochs=4, maxlen=200, max_learners=20000, seed=0, need=None):
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)

    def enc(seqs, cap=None):
        keys = list(seqs)
        if cap and len(keys) > cap:
            keys = list(rng.choice(np.array(keys, dtype=object), cap, replace=False))
        X, Y = [], []
        for u in keys:
            s = seqs[u]
            xi = (s + 1).tolist()                              # category code + 1; 0 = padding
            yi = idx[s].tolist()
            for st in range(0, max(1, len(s) - 1), maxlen):
                xx = xi[st:st + maxlen]; yy = yi[st + 1:st + maxlen + 1]
                if len(yy) < 1:
                    continue
                xx = xx[:len(yy)]
                X.append(xx); Y.append(yy)
        return X, Y

    def pad(batch, val):
        L = max(len(b) for b in batch)
        return torch.tensor([b + [val] * (L - len(b)) for b in batch])

    X, Y = enc(train_seqs, max_learners)
    m = GRUNext(int(vocab_index), n, seed=seed)
    opt = torch.optim.Adam(m.params, lr=2e-3)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    Xv, Yv = enc(val_seqs, 3000)
    best, best_state = None, None
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for b in range(0, len(order), 64):
            ix = order[b:b + 64]
            xb = pad([X[i] for i in ix], 0); yb = pad([Y[i] for i in ix], -1)
            opt.zero_grad(); lo = lossf(m.forward(xb).reshape(-1, n), yb.reshape(-1)); lo.backward(); opt.step()
        with torch.no_grad():
            tot = cnt = 0.0
            for b in range(0, len(Xv), 128):
                xb = pad(Xv[b:b + 128], 0); yb = pad(Yv[b:b + 128], -1)
                lo = torch.nn.functional.cross_entropy(m.forward(xb).reshape(-1, n), yb.reshape(-1), ignore_index=-1, reduction="sum")
                tot += float(lo); cnt += float((yb >= 0).sum())
        vl = tot / max(cnt, 1)
        print(f"  gru epoch {ep}: val loss {vl:.4f}", flush=True)
        if best is None or vl < best:
            best = vl; best_state = [p.detach().clone() for p in m.params]
    with torch.no_grad():
        for p, q in zip(m.params, best_state):
            p.copy_(q)
    # scores for every position of every test sequence (logits before the next run)
    out = {}
    with torch.no_grad():
        for u, s in test_seqs.items():
            if need is not None and u not in need:
                continue
            xi = (s + 1).tolist()
            h = None; logits = []
            for st in range(0, len(xi), maxlen):
                e = m.emb(torch.tensor([xi[st:st + maxlen]]))
                o, h = m.rnn(e, h)
                logits.append(m.out(o)[0].numpy())
            full = np.concatenate(logits, 0)
            if need is not None:
                pos = sorted(i for i, _ in need.get(u, []))
                out[u] = dict(zip(pos, full[pos])) if pos else {}
            else:
                out[u] = full
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--level", default="kc")
    ap.add_argument("--construction", default="copractice", choices=["copractice", "transition"])
    ap.add_argument("--max-test-learners", type=int, default=2000)
    ap.add_argument("--gru-epochs", type=int, default=6)
    ap.add_argument("--only-suggested", default=None, choices=[None, "unsuggested"],
                    help="Junyi: score only test events whose next attempt was not suggested by the platform")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = f"{a.release}_{a.level}_{a.construction}" + (f"_{a.only_suggested}" if a.only_suggested else "")
    out = a.out or f"results/v4/nextconcept_{tag}.json"
    df = load_any(a.release, a.level, keep_suggested=bool(a.only_suggested))
    cfg = ConstructionConfig()
    sugg = df[["user_id", "order", "suggested"]] if "suggested" in df.columns else None
    d = preprocess(df.drop(columns=[c for c in ("suggested",) if c in df.columns]), cfg)
    del df
    import gc; gc.collect()
    if not isinstance(d["kc"].dtype, pd.CategoricalDtype):
        d["kc"] = d["kc"].astype(str).astype("category")
    users = list(d["user_id"].unique()); spl = {u: split_id(u) for u in users}
    train_u = [u for u in users if spl[u] == "train"]
    G = (build_fuzzy_graph(d[d.user_id.isin(set(train_u))], cfg, preprocessed=True)[0] if a.construction == "copractice"
         else build_transition_graph(d[d.user_id.isin(set(train_u))], TransitionConfig(), preprocessed=True)[0])
    gc.collect()
    lab2node = {lab: i for i, lab in enumerate(G.labels)}; n = G.n
    cats = np.asarray(d["kc"].cat.categories.astype(str))
    idx = np.array([lab2node.get(c, -1) for c in cats], dtype=np.int64)     # category code -> node
    test_u = sorted(u for u in users if spl[u] == "test")[: a.max_test_learners]
    val_u = sorted(u for u in users if spl[u] == "val")
    tr_seqs = runs_by_learner(d, train_u); te_seqs = runs_by_learner(d, test_u); va_seqs = runs_by_learner(d, val_u)
    ev = events(te_seqs, idx)
    if a.only_suggested and sugg is not None:
        # keep events whose next run starts with an attempt the platform did not suggest
        s2 = d[d.user_id.isin(set(test_u))].merge(sugg, on=["user_id", "order"], how="left")
        flags = {}
        for u, g in s2.groupby("user_id", sort=False):
            k = g["kc"].cat.codes.to_numpy(); f = g["suggested"].fillna(False).to_numpy()
            keep = np.ones(k.size, dtype=bool); keep[1:] = k[1:] != k[:-1]
            flags[u] = f[keep]
        ev = [(u, i) for (u, i) in ev if not flags[u][i + 1]]
    src = np.array([idx[te_seqs[u][i]] for u, i in ev], dtype=np.int64); tgt = np.array([idx[te_seqs[u][i + 1]] for u, i in ev], dtype=np.int64)
    del d; gc.collect()
    uid = np.array([u for u, _ in ev], dtype=object)
    print(f"{tag}: graph {G.summary()}; test events {len(ev)} / learners {len(set(uid))}", flush=True)
    res = {"release": a.release, "level": a.level, "construction": a.construction, "graph": G.summary(),
           "n_nodes": n, "n_test_events": len(ev), "n_test_learners": int(len(set(uid))), "scores": {}, "paired": {}}
    per = {}

    def put(name, rows):
        rr, rra = rows
        r, per[name] = summarise(rr, rra, uid); res["scores"][name] = r
        print(f"  {name}: {r['mrr_learner_mean']:.4f}", flush=True)

    def by_source(M):
        rr = np.empty(len(ev)); rra = np.empty(len(ev))
        for b in range(0, len(ev), 20000):
            sl = slice(b, b + 20000); x, y = expected_rr(M[src[sl]], src[sl], tgt[sl]); rr[sl] = x; rra[sl] = y
        return rr, rra

    A = G.adjacency().toarray(); put("direct_edge", by_source(A))
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    Dh = shortest_path(csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(n, n)), directed=True, unweighted=True)
    put("hop_distance", by_source(np.where(np.isfinite(Dh), 1.0 / (1.0 + Dh), 0.0)))
    # validation-selected operator (as in Table 6)
    va_ev = events(va_seqs, idx)
    vsrc = np.array([idx[va_seqs[u][i]] for u, i in va_ev], dtype=np.int64); vtgt = np.array([idx[va_seqs[u][i + 1]] for u, i in va_ev], dtype=np.int64)
    best_k, best_v, mats = None, -1, {}
    for key, (name, p) in OPERATORS.items():
        S, _ = strength_matrix(G, name, p)
        mats[key] = S
        v = float(np.mean(np.concatenate([expected_rr(S[vsrc[b:b + 20000]], vsrc[b:b + 20000], vtgt[b:b + 20000])[0]
                                          for b in range(0, len(va_ev), 20000)]))) if len(va_ev) else 0.0
        if v > best_v:
            best_k, best_v = key, v
    put("product", by_source(mats["product"]))
    put("selected_operator", by_source(mats[best_k])); res["selected_operator"] = best_k
    # popularity and bigram on training learners
    pop = np.zeros(n); Bm = np.zeros((n, n)); M2 = defaultdict(lambda: defaultdict(float))   # sparse second-order counts
    for s in tr_seqs.values():
        ii = idx[s]
        for c in ii[ii >= 0]:
            pop[c] += 1
        for i in range(len(ii) - 1):
            if ii[i] >= 0 and ii[i + 1] >= 0:
                Bm[ii[i], ii[i + 1]] += 1
                if i >= 1:
                    M2[(ii[i - 1], ii[i])][ii[i + 1]] += 1
    put("popularity", by_source(np.tile(pop, (n, 1))))
    put("bigram", by_source(Bm))
    # recency and second-order Markov depend on the learner's history
    R = np.zeros((len(ev), n), dtype=np.float32); MK = np.zeros((len(ev), n))
    by_u = defaultdict(list)
    for e, (u, i) in enumerate(ev):
        by_u[u].append((i, e))
    for u, lst in by_u.items():                      # incremental: O(length x nodes) per learner
        s = te_seqs[u]; codes = idx[s]
        last = np.zeros(n, dtype=np.float32); j = 0
        for i, e in sorted(lst):
            while j <= i:
                if codes[j] >= 0:
                    last[codes[j]] = j + 1
                j += 1
            R[e] = last
            p = codes[i - 1] if i >= 1 else -1
            MK[e] = 1e-3 * Bm[src[e]] + 1e-6 * pop
            if p >= 0 and (p, src[e]) in M2:
                for c, v in M2[(p, src[e])].items():
                    MK[e, c] += v
    def chunked(M):
        rr = np.empty(len(ev)); rra = np.empty(len(ev))
        for b in range(0, len(ev), 20000):
            sl = slice(b, b + 20000); x, y = expected_rr(M[sl], src[sl], tgt[sl]); rr[sl] = x; rra[sl] = y
        return rr, rra
    put("recency", chunked(R)); del R
    put("markov2_backoff", chunked(MK)); del MK
    # GRU next-run model
    vocab = None
    t0 = time.perf_counter()
    logits, vl = gru_scores(tr_seqs, va_seqs, te_seqs, len(cats) + 1, idx, n, epochs=a.gru_epochs, need=by_u)
    Sg = np.stack([logits[u][i] for u, i in ev]) if ev else np.zeros((0, n))
    put("gru", chunked(Sg)); res["gru_val_loss"] = vl; res["gru_time_s"] = time.perf_counter() - t0
    ps = {}
    for base in ("popularity", "recency", "bigram", "markov2_backoff", "gru"):
        c = paired(per[base], per["selected_operator"]); res["paired"][f"{base}-selected_operator"] = c; ps[f"{base}-selected_operator"] = c["p"]
    c = paired(per["selected_operator"], per["hop_distance"]); res["paired"]["selected_operator-hop_distance"] = c; ps["selected_operator-hop_distance"] = c["p"]
    for k, v in holm(ps).items():
        res["paired"][k]["p_holm"] = v
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
