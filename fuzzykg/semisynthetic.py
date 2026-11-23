"""Semi-synthetic learner logs with a known prerequisite structure (release 5.0.0, Section 7).

A world has K concepts in B communities, a prerequisite DAG with graded strengths w in (0, 1],
and N learners. Prerequisites of a concept are drawn among the concepts before it in a global
order, inside its community with probability ``p_comm`` and outside otherwise. A learner covers
``span`` consecutive communities; inside the span it practises concepts in an order that follows
the DAG (a concept is eligible once its prerequisites have been attempted), with probability
``noise`` of picking a random concept of the current community instead, and with probability
``suggest`` the platform suggests a child of the concept just practised (the analogue of the
Junyi map-driven suggestions). An attempt is correct with probability

    sigmoid(theta + 2.0 * mastered - 1.0 - 1.5 * sum_{p unmastered} w(p, c)),

and after each attempt the concept becomes mastered with probability
``learn * prod_{p unmastered} (1 - w(p, c))``: mastery is gated by the prerequisites. A learner
leaves a concept once it is mastered or after ``max_tries`` attempts. With probability ``interleave`` a step is preceded by one review attempt on
a concept of the span practised so far (spiral review), which mixes communities in the log.

The log has the columns user_id, kc, correct, order of fuzzykg.construction, so both
constructions of Section 4 run on it unchanged. The ground truth returned with it is the
community of every concept and the DAG with its strengths.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@dataclass
class World:
    K: int = 120
    B: int = 8
    p_comm: float = 0.85
    parents_max: int = 3
    noise: float = 0.2
    suggest: float = 0.0
    interleave: float = 0.0
    N: int = 1500
    span: int = 3
    learn: float = 0.35
    max_tries: int = 6
    seed: int = 0
    delta: float = 0.0   # community information in the strengths (release 5.1.0): a prerequisite inside the
                         # community of its target has strength U(0.3 + delta, 1), one outside U(0.3, 1 - delta)


def make_dag(w: World, rng):
    comm = np.repeat(np.arange(w.B), w.K // w.B)
    K = comm.size
    # global order: communities in order, concepts in order within a community
    par_src, par_dst, par_w = [], [], []
    for c in range(K):
        pos_in = np.flatnonzero((comm == comm[c]) & (np.arange(K) < c))
        pos_out = np.flatnonzero((comm < comm[c]))
        k = rng.integers(1, w.parents_max + 1)
        chosen = set()
        for _ in range(k):
            if pos_in.size and (rng.random() < w.p_comm or pos_out.size == 0):
                p = int(rng.choice(pos_in))
            elif pos_out.size:
                p = int(rng.choice(pos_out[-3 * (K // w.B):]))
            else:
                continue
            chosen.add(p)
        for p in chosen:
            inside = comm[p] == comm[c]
            lo, hi = (0.3 + w.delta, 1.0) if inside else (0.3, 1.0 - w.delta)
            par_src.append(p); par_dst.append(c); par_w.append(float(rng.uniform(lo, hi)))
    return comm, np.array(par_src, int), np.array(par_dst, int), np.array(par_w)


@njit(cache=True)
def _simulate(K, comm, B, span, pptr, pidx, pw, cptr, cidx, N, noise, suggest, interleave, learn, max_tries, seed):
    np.random.seed(seed)
    cap = N * K * 3
    U = np.empty(cap, np.int64); C = np.empty(cap, np.int64); Y = np.empty(cap, np.int8)
    t = 0
    for u in range(N):
        theta = np.random.normal()
        start = np.random.randint(0, B - span + 1)
        mastered = np.zeros(K, np.bool_); attempted = np.zeros(K, np.bool_)
        last = -1
        for b in range(start, start + span):
            members = np.flatnonzero(comm == b)
            remaining = members.size
            done = np.zeros(K, np.bool_)
            while remaining > 0:
                if interleave > 0 and np.random.random() < interleave:
                    # spiral review: one attempt on a concept of the span practised so far
                    rev = np.flatnonzero(attempted & (comm >= start) & (comm <= b))
                    if rev.size:
                        r = rev[np.random.randint(0, rev.size)]
                        pen = 0.0
                        for j in range(pptr[r], pptr[r + 1]):
                            if not mastered[pidx[j]]:
                                pen += pw[j]
                        z = theta + 2.0 * mastered[r] - 1.0 - 1.5 * pen
                        U[t] = u; C[t] = r; Y[t] = np.random.random() < 1.0 / (1.0 + np.exp(-z)); t += 1
                nxt = -1
                if last >= 0 and np.random.random() < suggest:
                    # platform suggests a not yet finished child of the last concept in this community
                    for j in range(cptr[last], cptr[last + 1]):
                        ch = cidx[j]
                        if comm[ch] == b and not done[ch]:
                            nxt = ch
                            break
                if nxt < 0 and np.random.random() < noise:
                    cand = members[~done[members]]
                    nxt = cand[np.random.randint(0, cand.size)]
                if nxt < 0:
                    # first eligible concept in the global order: all prerequisites attempted
                    for c in members:
                        if done[c]:
                            continue
                        ok = True
                        for j in range(pptr[c], pptr[c + 1]):
                            if not attempted[pidx[j]]:
                                ok = False
                                break
                        if ok:
                            nxt = c
                            break
                    if nxt < 0:
                        cand = members[~done[members]]
                        nxt = cand[0]
                c = nxt
                for _ in range(max_tries):
                    pen = 0.0; gate = 1.0
                    for j in range(pptr[c], pptr[c + 1]):
                        if not mastered[pidx[j]]:
                            pen += pw[j]; gate *= (1.0 - pw[j])
                    z = theta + 2.0 * mastered[c] - 1.0 - 1.5 * pen
                    y = np.random.random() < 1.0 / (1.0 + np.exp(-z))
                    U[t] = u; C[t] = c; Y[t] = y; t += 1
                    attempted[c] = True
                    if not mastered[c] and np.random.random() < learn * gate:
                        mastered[c] = True
                    if mastered[c] and y:
                        break
                done[c] = True; remaining -= 1; last = c
    return U[:t], C[:t], Y[:t]


def simulate(w: World):
    rng = np.random.default_rng(w.seed)
    comm, ps, pd_, pw = make_dag(w, rng)
    K = comm.size
    o = np.argsort(pd_, kind="stable")
    pptr = np.zeros(K + 1, np.int64); np.add.at(pptr, pd_ + 1, 1); pptr = np.cumsum(pptr)
    pidx, pww = ps[o], pw[o]
    o2 = np.argsort(ps, kind="stable")
    cptr = np.zeros(K + 1, np.int64); np.add.at(cptr, ps + 1, 1); cptr = np.cumsum(cptr)
    cidx = pd_[o2]
    U, C, Y = _simulate(K, comm, w.B, w.span, pptr, pidx, pww, cptr, cidx, w.N, w.noise, w.suggest, w.interleave,
                        w.learn, w.max_tries, w.seed)
    log = pd.DataFrame({"user_id": U.astype(str), "kc": C, "correct": Y.astype(int), "order": np.arange(U.size)})
    truth = {"community": comm, "dag_src": ps, "dag_dst": pd_, "dag_w": pw, "world": asdict(w)}
    return log, truth


def true_prereq_score(truth, K):
    """Graded prerequisite score of every ordered pair (b, a): the strongest product of DAG
    strengths over directed DAG paths from b to a (0 when b is not an ancestor of a)."""
    S = np.zeros((K, K))
    order = np.arange(K)  # the DAG respects the index order
    par = [[] for _ in range(K)]
    for s, d, w in zip(truth["dag_src"], truth["dag_dst"], truth["dag_w"]):
        par[d].append((s, w))
    for a in order:
        for s, w in par[a]:
            S[s, a] = max(S[s, a], w)
            S[:, a] = np.maximum(S[:, a], S[:, s] * w)
    return S
