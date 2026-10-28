"""Constructing a fuzzy knowledge graph from interaction logs (Section 7.1).

Two constructions are implemented: the co-practice construction below
(``build_fuzzy_graph``) and the order-preserving transition construction
(``build_transition_graph``).

Three signals in [0,1] per ordered pair (c_a, c_b):

* temporal precedence  (Eq. 14, implemented as the directional share of
  within-horizon co-occurrences, N(a -> b) / (N(a -> b) + N(b -> a)));
* empirical mastery gain  (difference in mean correctness on c_b between
  learners with prior partial mastery of c_a and learners without, mapped
  into [0,1] by a logistic squash of slope 5);
* optional taxonomy prior  (0.9 if an expert curriculum places c_b downstream
  of c_a, else 0).

They are aggregated by an OWA operator (Eq. 15) with w = (0.5, 0.3, 0.2),
orness 0.65, and edges with mu_E < tau_E = 0.2 are pruned.

Input format
------------
A pandas DataFrame with columns ``user_id``, ``kc``, ``correct`` (0/1) and
``order`` (an integer giving the release's log order within a learner).
``load_assistments`` in ``data/loaders.py`` produces this from the public
releases.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .graph import FuzzyGraph
from .tnorms import owa


@dataclass
class ConstructionConfig:
    horizon: int = 10               # precedence window (interactions)
    min_support: int = 20           # minimum within-horizon co-occurrences
    owa_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    tau_E: float = 0.2              # pruning threshold
    gain_slope: float = 5.0         # logistic squash slope
    mastery_threshold: float = 0.5  # "partial mastery of c_a": prior accuracy on c_a >= this
    min_interactions: int = 15      # learners with fewer retained interactions are dropped
    taxonomy: Optional[Set[Tuple[str, str]]] = None   # expert (a, b) pairs, b downstream of a


def preprocess(df: pd.DataFrame, cfg: ConstructionConfig) -> pd.DataFrame:
    """Stated preprocessing: drop missing KC, drop responses outside {0,1},
    drop duplicate rows, order by log order, drop short learners."""
    d = df.dropna(subset=["kc"]).copy()
    d = d[d["correct"].isin([0, 1])]
    d = d.drop_duplicates()
    d = d.sort_values(["user_id", "order"], kind="stable")
    counts = d.groupby("user_id").size()
    keep = counts[counts >= cfg.min_interactions].index
    return d[d["user_id"].isin(keep)].reset_index(drop=True)


def precedence_counts(d: pd.DataFrame, horizon: int) -> Dict[Tuple[str, str], int]:
    """N(a -> b): number of (t, t') with t < t' <= t + horizon, kc_t = a, kc_t' = b, a != b."""
    N: Dict[Tuple[str, str], int] = defaultdict(int)
    for _, seq in d.groupby("user_id", sort=False)["kc"]:
        kcs = seq.tolist()
        L = len(kcs)
        for i in range(L):
            a = kcs[i]
            for j in range(i + 1, min(L, i + horizon + 1)):
                b = kcs[j]
                if a != b:
                    N[(a, b)] += 1
    return N


def mastery_gain(d: pd.DataFrame, pairs, cfg: ConstructionConfig) -> Dict[Tuple[str, str], float]:
    """gain(a, b) = mean correct on b | prior partial mastery of a  -  mean correct on b | no prior a,
    squashed by sigma(slope * gain)."""
    # running accuracy per (user, kc) before each attempt
    d = d.copy()
    d["_cum"] = d.groupby(["user_id", "kc"])["correct"].cumsum() - d["correct"]
    d["_n"] = d.groupby(["user_id", "kc"]).cumcount()
    per_user: Dict[str, Dict[str, list]] = {}
    for uid, g in d.groupby("user_id", sort=False):
        per_user[uid] = {"kc": g["kc"].tolist(), "correct": g["correct"].tolist()}
    wanted = set(pairs)
    A_needed = {a for a, _ in wanted}
    with_m = defaultdict(lambda: [0, 0])
    without = defaultdict(lambda: [0, 0])
    for uid, rec in per_user.items():
        kcs, cor = rec["kc"], rec["correct"]
        seen_n: Dict[str, int] = defaultdict(int)
        seen_c: Dict[str, int] = defaultdict(int)
        mastered: Set[str] = set()
        for k, c in zip(kcs, cor):
            # record attempt on b = k against every candidate a
            for a in A_needed:
                if (a, k) not in wanted:
                    continue
                if a in mastered:
                    with_m[(a, k)][0] += c; with_m[(a, k)][1] += 1
                elif seen_n[a] == 0:
                    without[(a, k)][0] += c; without[(a, k)][1] += 1
            seen_n[k] += 1; seen_c[k] += c
            if seen_n[k] > 0 and seen_c[k] / seen_n[k] >= cfg.mastery_threshold:
                mastered.add(k)
    out = {}
    for p in wanted:
        cw, nw = with_m[p]
        co, no = without[p]
        if nw == 0 or no == 0:
            gain = 0.0
        else:
            gain = cw / nw - co / no
        out[p] = float(1.0 / (1.0 + np.exp(-cfg.gain_slope * gain)))
    return out


def _signals_fast(d: pd.DataFrame, cfg: ConstructionConfig):
    """Precedence share and mastery gain for every pair with enough support,
    computed with the compiled kernels of ``fastcount``.  Same definitions as
    ``precedence_counts`` and ``mastery_gain``."""
    from .fastcount import encode, precedence_matrix, mastery_counts
    ptr, kc, cor, labels = encode(d)
    K = len(labels)
    N = precedence_matrix(ptr, kc, K, cfg.horizon)
    tot = N + N.T
    keep = (tot >= cfg.min_support) & (N > 0)
    np.fill_diagonal(keep, False)
    A, B = np.nonzero(keep)
    prec = N[A, B] / tot[A, B]
    # CSR of wanted a's per b
    order = np.argsort(B, kind="stable")
    wanted_a = A[order].astype(np.int64)
    wanted_ptr = np.zeros(K + 1, dtype=np.int64)
    np.add.at(wanted_ptr, B + 1, 1)
    wanted_ptr = np.cumsum(wanted_ptr)
    wc, wn, oc, on = mastery_counts(ptr, kc, cor, K, wanted_ptr, wanted_a, cfg.mastery_threshold)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where((wn[A, B] > 0) & (on[A, B] > 0), wc[A, B] / np.maximum(wn[A, B], 1) - oc[A, B] / np.maximum(on[A, B], 1), 0.0)
    gain = 1.0 / (1.0 + np.exp(-cfg.gain_slope * gain))
    return labels, A, B, prec, gain


def _assemble(tab: pd.DataFrame) -> FuzzyGraph:
    labels = sorted(set(tab["src"]) | set(tab["dst"]), key=str)
    idx = {k: i for i, k in enumerate(labels)}
    return FuzzyGraph.from_edges(
        ((idx[a], idx[b], m) for a, b, m in zip(tab["src"], tab["dst"], tab["mu"])),
        n=len(labels), labels=[str(x) for x in labels],
    )


def build_fuzzy_graph(df: pd.DataFrame, cfg: ConstructionConfig = ConstructionConfig(),
                      fast: Optional[bool] = None, preprocessed: bool = False) -> Tuple[FuzzyGraph, pd.DataFrame]:
    """Run Section 7.1 (the co-practice construction) end to end.  Returns the
    graph and an edge table with the three signals and the aggregated membership.
    ``fast`` selects the compiled kernels (default: when numba is installed)."""
    from .fastcount import HAVE_NUMBA
    d = df if preprocessed else preprocess(df, cfg)
    if fast is None:
        # the compiled kernels hold K x K count matrices; fall back to the sparse
        # dictionaries when there are too many KCs (item-level logs)
        fast = HAVE_NUMBA and d["kc"].nunique() <= 4000
    rows = []
    if fast:
        labels, A, B, prec, gain = _signals_fast(d, cfg)
        for a, b, p, g in zip(A, B, prec, gain):
            la, lb = labels[a], labels[b]
            tax = 0.9 if (cfg.taxonomy is not None and (la, lb) in cfg.taxonomy) else 0.0
            rows.append((la, lb, float(p), float(g), tax, owa([p, g, tax], cfg.owa_weights)))
    else:
        N = precedence_counts(d, cfg.horizon)
        pairs = [(a, b) for (a, b), n in N.items() if n + N.get((b, a), 0) >= cfg.min_support]
        prec = {(a, b): N[(a, b)] / (N[(a, b)] + N.get((b, a), 0)) for (a, b) in pairs}
        gain = mastery_gain(d, pairs, cfg)
        for (a, b) in pairs:
            tax = 0.9 if (cfg.taxonomy is not None and (a, b) in cfg.taxonomy) else 0.0
            rows.append((a, b, prec[(a, b)], gain[(a, b)], tax, owa([prec[(a, b)], gain[(a, b)], tax], cfg.owa_weights)))
    tab = pd.DataFrame(rows, columns=["src", "dst", "precedence", "gain", "taxonomy", "mu"])
    tab = tab[tab["mu"] >= cfg.tau_E].reset_index(drop=True)
    return _assemble(tab), tab


@dataclass
class TransitionConfig:
    """The order-preserving construction (Section 7.1, second construction)."""
    min_support: int = 20           # minimum number of moves between a and b in either direction
    tau_E: float = 0.2              # pruning threshold
    min_interactions: int = 15


def build_transition_graph(df: pd.DataFrame, cfg: TransitionConfig = TransitionConfig(),
                           preprocessed: bool = False) -> Tuple[FuzzyGraph, pd.DataFrame]:
    """Order-preserving construction.  Repeats of a KC are collapsed into runs and
    T(a -> b) counts moves from a run of a to the next run, of b.  For a pair with
    T(a -> b) + T(b -> a) >= min_support, two degrees in [0,1] are formed:

        asym(a, b) = max(0, 2 T(a->b) / (T(a->b) + T(b->a)) - 1)   (directional dominance)
        pref(a, b) = T(a->b) / max_c T(a->c)                        (b is a's preferred successor)

    and the membership is their minimum, mu = min(asym, pref): an edge a -> b is a
    prerequisite to the degree that learners move from a to b rather than back
    AND that b is where learners go after a.  Only the dominant direction of a pair
    can have mu > 0, so the graph has no reciprocal edges.  Edges with mu < tau_E
    are pruned."""
    from .fastcount import encode
    d = df if preprocessed else preprocess(df, ConstructionConfig(min_interactions=cfg.min_interactions))
    ptr, kc, _, labels = encode(d)
    K = len(labels)
    # moves between consecutive runs of the same learner (sparse counting)
    users = np.repeat(np.arange(ptr.size - 1), np.diff(ptr))
    new_run = np.ones(kc.size, dtype=bool)
    new_run[1:] = (kc[1:] != kc[:-1]) | (users[1:] != users[:-1])
    rk, ru = kc[new_run], users[new_run]
    same = ru[1:] == ru[:-1]
    code = rk[:-1][same].astype(np.int64) * K + rk[1:][same]
    uniq, cnt = np.unique(code, return_counts=True)
    Tab = dict(zip(uniq.tolist(), cnt.tolist()))
    a_all, b_all = uniq // K, uniq % K
    rowmax = np.zeros(K); np.maximum.at(rowmax, a_all, cnt)
    back = np.array([Tab.get(int(b) * K + int(a), 0) for a, b in zip(a_all, b_all)], dtype=float)
    fwd = cnt.astype(float); tot = fwd + back
    keep = (tot >= cfg.min_support) & (fwd > back)
    A, B = a_all[keep], b_all[keep]
    asym = 2.0 * fwd[keep] / tot[keep] - 1.0
    pref = fwd[keep] / rowmax[A]
    mu = np.minimum(asym, pref)
    T_fwd, T_back = fwd[keep], back[keep]
    tab = pd.DataFrame({"src": labels[A], "dst": labels[B], "asym": asym, "pref": pref,
                        "moves": T_fwd.astype(int), "moves_back": T_back.astype(int), "mu": mu})
    tab = tab[tab["mu"] >= cfg.tau_E].reset_index(drop=True)
    return _assemble(tab), tab
