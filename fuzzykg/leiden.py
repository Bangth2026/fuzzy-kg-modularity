"""Fuzzy Leiden: the Leiden procedure of Traag et al. with Q_F as objective.

Because Q_F (Eq. 7) is the directed Leicht-Newman functional evaluated on the
memberships, the ``RBConfigurationVertexPartition`` quality of ``leidenalg``
on a directed weighted igraph *is* Q_F, and the local-moving / refinement /
aggregation phases are those of [7].  ``fuzzy_leiden`` wraps that and
returns the partition together with the two computational checks used in
Section 8:

* Proposition 1   -- every community induces a weakly connected subgraph of
                     the support graph (``check_weak_connectivity``);
* Proposition 2(iii) -- Q_F is non-decreasing over successive optimisation
                     iterations (``check_monotone``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .graph import FuzzyGraph
from .modularity import fuzzy_modularity


@dataclass
class LeidenResult:
    membership: np.ndarray
    Q_F: float
    gamma: float
    seed: int
    n_communities: int
    sizes: List[int]
    trajectory: List[float]          # Q_F after each optimisation iteration
    connected_violations: int        # Proposition 1 counter-examples
    monotone_violations: int         # Proposition 2(iii) counter-examples


def check_weak_connectivity(G: FuzzyGraph, membership) -> int:
    """Number of communities that do NOT induce a weakly connected subgraph."""
    memb = np.asarray(membership)
    A = G.symmetrised()
    bad = 0
    for k in np.unique(memb):
        idx = np.where(memb == k)[0]
        if idx.size <= 1:
            continue
        sub = A[idx][:, idx]
        ncc, _ = connected_components(sub, directed=False)
        if ncc > 1:
            bad += 1
    return bad


def check_monotone(trajectory, tol: float = 1e-9) -> int:
    """Number of successive decreases of Q_F larger than ``tol``."""
    d = np.diff(np.asarray(trajectory, dtype=float))
    return int(np.sum(d < -tol))


def fuzzy_leiden(
    G: FuzzyGraph,
    gamma: float = 1.0,
    seed: int = 0,
    n_iterations: int = 10,
    until_convergence: bool = True,
) -> LeidenResult:
    """Run Leiden with Q_F as the local-moving objective.

    Parameters
    ----------
    gamma : resolution parameter of Eq. (7).
    n_iterations : number of recorded optimisation iterations (Table 4 uses
        10, Table 5 uses 12).  Each call to ``optimise_partition`` is one
        local-moving + refinement + aggregation sweep.
    until_convergence : after the recorded iterations, keep iterating until
        no improvement (``n_iterations=-1`` in leidenalg).
    """
    import leidenalg as la

    g = G.to_igraph()
    part = la.RBConfigurationVertexPartition(g, weights="weight", resolution_parameter=gamma)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed))

    traj = [fuzzy_modularity(G, part.membership, gamma)]
    for _ in range(n_iterations):
        opt.optimise_partition(part, n_iterations=1)
        traj.append(fuzzy_modularity(G, part.membership, gamma))
    if until_convergence:
        opt.optimise_partition(part, n_iterations=-1)
        traj.append(fuzzy_modularity(G, part.membership, gamma))

    memb = np.asarray(part.membership, dtype=np.int64)
    sizes = sorted(np.bincount(memb).tolist(), reverse=True)
    QF = fuzzy_modularity(G, memb, gamma)
    # sanity: leidenalg's (unnormalised) quality divided by W must agree with Eq. (8)
    assert abs(part.quality() / G.W - QF) < 1e-8, (part.quality() / G.W, QF)
    return LeidenResult(
        membership=memb,
        Q_F=QF,
        gamma=gamma,
        seed=seed,
        n_communities=len(sizes),
        sizes=sizes,
        trajectory=traj,
        connected_violations=check_weak_connectivity(G, memb),
        monotone_violations=check_monotone(traj),
    )


def best_of_seeds(G: FuzzyGraph, gamma: float, seeds=range(10), **kw) -> LeidenResult:
    runs = [fuzzy_leiden(G, gamma, seed=s, **kw) for s in seeds]
    return max(runs, key=lambda r: r.Q_F)


def adjusted_rand_index(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score  # optional dependency

    return float(adjusted_rand_score(a, b))
