"""Level-set multislice modularity (Section 5.5).

A fuzzy graph has a canonical family of crisp level sets E_alpha.  Instead of
optimising one of them (the binarise-then-partition shortcut) or the fuzzy
mass itself (Q_F), the level-set objective

    Qbar(Pi) = sum_l w_l Q_{alpha_l}(Pi; gamma_l),   sum_l w_l = 1,

with Q_alpha the directed modularity of the level set E_alpha under its OWN
normalisation (total crisp mass |E_alpha|), is optimised for a single
partition Pi shared by every level.  This is the multislice modularity of
Mucha et al. (2010) with the level sets as slices, the identity coupling of a
node to itself across slices, and a shared partition, so that the coupling
term is constant and the objective reduces to the weighted average above.
Two weightings are implemented: ``weighting="mass"`` (default) takes
w_l proportional to |E_{alpha_l}|, the layer-cake weighting under which the
sum approximates the integral of |E_alpha| Q_alpha over alpha; ``"equal"``
takes w_l = 1/L, the plain average over levels.

``multislice_leiden`` optimises Qbar with leidenalg's multiplex optimiser
(each level set is one layer, layer weight w_l/|E_alpha|), and
``level_set_profile`` returns Q_alpha(Pi) on every level, from which the
binarisation cost of a single-cut partition is read off as

    cost(alpha_0) = Qbar(Pi*) - Qbar(Pi_{alpha_0}),

the drop in the level-set objective when the partition of one level is used
for all of them.  ``persistence`` tracks the communities of a crisp detector
along the filtration and returns each community's lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .graph import FuzzyGraph
from .modularity import fuzzy_modularity


def level_grid(G: FuzzyGraph, levels: Optional[Sequence[float]] = None, n_levels: int = 8,
               quantile: bool = True) -> np.ndarray:
    """Default alpha-grid: quantiles of the membership distribution (scale-free),
    excluding the top one so that no level set is empty."""
    if levels is not None:
        return np.asarray(sorted(levels), dtype=float)
    if quantile:
        qs = np.linspace(0, 1, n_levels + 1)[:-1]
        return np.unique(np.quantile(G.mu, qs))
    return np.linspace(G.mu.min(), G.mu.max(), n_levels + 1)[:-1]


def level_set_modularity(G: FuzzyGraph, membership, alpha: float, gamma: float = 1.0) -> float:
    """Directed modularity of the crisp level set E_alpha under its own normalisation."""
    Ga = G.alpha_cut(alpha)
    return fuzzy_modularity(Ga, membership, gamma)


def level_weights(G: FuzzyGraph, alphas: Sequence[float], weighting: str = "mass") -> np.ndarray:
    sizes = np.array([(G.mu >= a).sum() for a in alphas], dtype=float)
    w = sizes if weighting == "mass" else np.ones(len(alphas))
    return w / w.sum()


def level_set_profile(G: FuzzyGraph, membership, alphas: Sequence[float],
                      gammas: Optional[Sequence[float]] = None) -> np.ndarray:
    gammas = [1.0] * len(alphas) if gammas is None else list(gammas)
    out = []
    for a, g in zip(alphas, gammas):
        try:
            out.append(level_set_modularity(G, membership, a, g))
        except ValueError:                      # empty level set
            out.append(0.0)
    return np.asarray(out)


def qbar(G: FuzzyGraph, membership, alphas: Sequence[float], gammas=None, weighting: str = "mass") -> float:
    w = level_weights(G, alphas, weighting)
    return float((w * level_set_profile(G, membership, alphas, gammas)).sum())


@dataclass
class MultisliceResult:
    membership: np.ndarray
    Qbar: float
    profile: np.ndarray          # Q_alpha(Pi) per level
    alphas: np.ndarray
    gammas: np.ndarray
    seed: int
    n_communities: int
    Q_F: float                   # the same partition evaluated on the fuzzy graph at gamma = 1
    weighting: str = "mass"


def multislice_leiden(G: FuzzyGraph, alphas: Optional[Sequence[float]] = None,
                      gammas: Optional[Sequence[float]] = None, seed: int = 0,
                      n_levels: int = 8, weighting: str = "mass") -> MultisliceResult:
    """Optimise Qbar over one partition shared by all level sets."""
    import igraph as ig
    import leidenalg as la

    alphas = level_grid(G, alphas, n_levels)
    gammas = np.ones(len(alphas)) if gammas is None else np.asarray(gammas, dtype=float)
    graphs, parts, weights = [], [], []
    wl = level_weights(G, alphas, weighting)
    for a, g, w in zip(alphas, gammas, wl):
        keep = G.mu >= a
        if not keep.any():
            continue
        gi = ig.Graph(n=G.n, edges=list(zip(G.src[keep].tolist(), G.dst[keep].tolist())), directed=True)
        gi.es["weight"] = [1.0] * int(keep.sum())
        graphs.append(gi)
        parts.append(la.RBConfigurationVertexPartition(gi, weights="weight", resolution_parameter=float(g)))
        weights.append(float(w) / float(keep.sum()))
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed))
    opt.optimise_partition_multiplex(parts, layer_weights=weights, n_iterations=-1)
    memb = np.asarray(parts[0].membership, dtype=np.int64)
    prof = level_set_profile(G, memb, alphas, gammas)
    # cross-check: the optimiser's weighted quality equals the mean of per-level modularities
    q_opt = sum(w * p.quality() for w, p in zip(weights, parts))
    q_bar = float((wl * prof).sum())
    assert abs(q_opt - q_bar) < 1e-8, (q_opt, q_bar)
    return MultisliceResult(membership=memb, Qbar=q_bar, profile=prof, alphas=alphas,
                            gammas=gammas, seed=seed, n_communities=int(np.unique(memb).size),
                            Q_F=fuzzy_modularity(G, memb, 1.0), weighting=weighting)


def best_multislice(G: FuzzyGraph, seeds=range(5), **kw) -> MultisliceResult:
    runs = [multislice_leiden(G, seed=s, **kw) for s in seeds]
    return max(runs, key=lambda r: r.Qbar)


def binarisation_cost(G: FuzzyGraph, pi_star, pi_alpha, alphas, gammas=None, weighting="mass") -> float:
    """cost(alpha_0) = Qbar(Pi*) - Qbar(Pi_alpha_0): what the single-cut partition
    gives up on the whole family of level sets."""
    return qbar(G, pi_star, alphas, gammas, weighting) - qbar(G, pi_alpha, alphas, gammas, weighting)


# ------------------------------------------------------------- persistence
def _match(prev: Dict[int, set], cur: Dict[int, set], jaccard_min: float = 0.5):
    """Match communities of successive levels by maximal Jaccard overlap."""
    out = {}
    for k, s in cur.items():
        best, bj = None, 0.0
        for kp, sp_ in prev.items():
            j = len(s & sp_) / len(s | sp_)
            if j > bj:
                best, bj = kp, j
        out[k] = best if bj >= jaccard_min else None
    return out


def persistence(G: FuzzyGraph, alphas: Sequence[float], seed: int = 0, gamma: float = 1.0,
                jaccard_min: float = 0.5) -> List[dict]:
    """Run crisp Leiden on each level set along the filtration (alpha increasing)
    and track every community's lifetime by Jaccard matching across successive
    levels.  Returns one record per track: birth level, death level, mass at
    birth, size at birth.  The mass of short-lived tracks that straddle a level
    alpha_0 is the persistence reading of the binarisation cost at alpha_0."""
    from .leiden import fuzzy_leiden

    alphas = list(alphas)
    tracks: List[dict] = []
    live: Dict[int, int] = {}       # community id at current level -> track index
    prev_sets: Dict[int, set] = {}
    for i, a in enumerate(alphas):
        try:
            Ga = G.alpha_cut(a)
        except ValueError:
            break
        r = fuzzy_leiden(Ga, gamma, seed=seed, n_iterations=0)
        cur_sets = {int(k): set(np.where(r.membership == k)[0].tolist()) for k in np.unique(r.membership)}
        cur_sets = {k: s for k, s in cur_sets.items() if len(s) > 1}
        match = _match(prev_sets, cur_sets, jaccard_min) if prev_sets else {k: None for k in cur_sets}
        new_live: Dict[int, int] = {}
        for k, s in cur_sets.items():
            m = match.get(k)
            if m is not None and m in live:
                t = live[m]
                tracks[t]["death"] = None
                tracks[t]["last_level"] = a
                new_live[k] = t
            else:
                idx = np.array(sorted(s))
                inside = np.isin(G.src, idx) & np.isin(G.dst, idx)
                tracks.append({"birth": a, "death": None, "last_level": a,
                               "size": len(s), "mass": float(G.mu[inside].sum())})
                new_live[k] = len(tracks) - 1
        for k, t in live.items():
            if t not in new_live.values():
                tracks[t]["death"] = a
        live, prev_sets = new_live, cur_sets
    return tracks


def persistence_cost(tracks: List[dict], alpha0: float, W: float, min_life: float) -> float:
    """Share of the fuzzy mass carried by tracks with lifetime < min_life that
    are alive at alpha0 (born at or before, dead after)."""
    mass = 0.0
    for t in tracks:
        death = t["death"] if t["death"] is not None else float("inf")
        life = death - t["birth"]
        if t["birth"] <= alpha0 < death and life < min_life:
            mass += t["mass"]
    return mass / W
