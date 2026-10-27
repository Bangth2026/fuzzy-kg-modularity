"""Synthetic fuzzy prerequisite graphs (Section 8.2).

Matched to the MOOCCubeX-CS structural statistics: density |E|/|C| ~ 1.54,
40 topics per 1000 nodes, edges within a topic with probability p_within and
across topics otherwise, directions constrained to a random curriculum order
(so the graph is acyclic), memberships ~ Beta(2.0, 1.2) clipped to [0.2, 1].
"""

from __future__ import annotations

import numpy as np

from .graph import FuzzyGraph


def synthetic_fuzzy_graph(
    n: int,
    p_within: float = 0.85,
    density: float = 1.54,
    topics_per_1000: int = 40,
    beta_a: float = 2.0,
    beta_b: float = 1.2,
    mu_min: float = 0.2,
    seed: int = 0,
) -> FuzzyGraph:
    rng = np.random.default_rng(seed)
    n_topics = max(2, int(round(topics_per_1000 * n / 1000)))
    topic = rng.integers(0, n_topics, size=n)
    order = rng.permutation(n)                      # curriculum order: rank of each node
    rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)
    members = [np.where(topic == t)[0] for t in range(n_topics)]
    m_target = int(round(density * n))
    edges = set()
    tries = 0
    while len(edges) < m_target and tries < 50 * m_target:
        tries += 1
        u = int(rng.integers(0, n))
        if rng.random() < p_within:
            pool = members[topic[u]]
        else:
            t2 = int(rng.integers(0, n_topics))
            while t2 == topic[u] and n_topics > 1:
                t2 = int(rng.integers(0, n_topics))
            pool = members[t2]
        if pool.size <= 1:
            continue
        v = int(rng.choice(pool))
        if v == u:
            continue
        a, b = (u, v) if rank[u] < rank[v] else (v, u)   # respect curriculum order
        edges.add((a, b))
    e = np.array(sorted(edges))
    mu = np.clip(rng.beta(beta_a, beta_b, size=len(e)), mu_min, 1.0)
    return FuzzyGraph.from_edges(zip(e[:, 0], e[:, 1], mu), n=n)


# ------------------------------------------------ planted partitions (Table 6)
def _beta_params(mean: float, sd: float):
    """Beta(a, b) with the given mean and standard deviation (sd clipped to the
    feasible range sd^2 < mean(1-mean))."""
    mean = float(np.clip(mean, 0.02, 0.98))
    vmax = mean * (1 - mean)
    var = min(sd ** 2, 0.95 * vmax)
    k = vmax / var - 1.0
    return mean * k, (1 - mean) * k


def planted_partition_fuzzy_graph(
    n: int = 1000,
    k: int = 20,
    out_degree: float = 8.0,
    p_in: float = 0.8,
    mu_mean: float = 0.6,
    mu_sd: float = 0.15,
    shift: float = 0.0,
    mu_min: float = 1e-3,
    seed: int = 0,
):
    """Directed planted-partition graph with controllable membership distribution.

    n nodes in k equal-size communities (the ground truth); each node emits
    ``out_degree`` edges on average, a fraction ``p_in`` of them inside its own
    community.  Memberships are Beta-distributed with mean ``mu_mean`` and
    standard deviation ``mu_sd``; ``shift`` in [0, 1) moves the mean of
    within-community edges up and of between-community edges down by
    ``shift * min(mu_mean, 1 - mu_mean)``, so that the memberships carry
    community information (shift > 0) or none (shift = 0).

    Returns (graph, ground-truth membership).
    """
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]
    rng.shuffle(truth)
    members = [np.where(truth == c)[0] for c in range(k)]
    m_target = int(round(out_degree * n))
    edges = {}
    tries = 0
    while len(edges) < m_target and tries < 50 * m_target:
        tries += 1
        u = int(rng.integers(0, n))
        inside = rng.random() < p_in
        if inside:
            pool = members[truth[u]]
        else:
            c2 = int(rng.integers(0, k))
            while c2 == truth[u]:
                c2 = int(rng.integers(0, k))
            pool = members[c2]
        v = int(rng.choice(pool))
        if v == u or (u, v) in edges:
            continue
        edges[(u, v)] = inside
    e = np.array(sorted(edges))
    inside = np.array([edges[tuple(x)] for x in e])
    d = shift * min(mu_mean, 1 - mu_mean)
    mu = np.empty(len(e))
    a_in, b_in = _beta_params(mu_mean + d, mu_sd)
    a_out, b_out = _beta_params(mu_mean - d, mu_sd)
    mu[inside] = rng.beta(a_in, b_in, size=int(inside.sum()))
    mu[~inside] = rng.beta(a_out, b_out, size=int((~inside).sum()))
    mu = np.clip(mu, mu_min, 1.0)
    return FuzzyGraph.from_edges(zip(e[:, 0], e[:, 1], mu), n=n), truth


# ------------------------------------------------ LFR-style directed fuzzy benchmark
def lfr_style_fuzzy_graph(
    n: int = 1000,
    mixing: float = 0.3,
    k_avg: float = 8.0,
    k_max: float = 40.0,
    tau_deg: float = 2.0,
    tau_size: float = 1.0,
    c_min: int = 10,
    c_max: int | None = None,
    mu_mean: float = 0.5,
    mu_sd: float = 0.15,
    shift: float = 0.0,
    mu_min: float = 1e-3,
    seed: int = 0,
):
    """Directed, weighted benchmark in the style of Lancichinetti–Fortunato [29]:
    community sizes from a power law with exponent ``tau_size`` on
    [c_min, c_max], node out-degrees from a power law with exponent ``tau_deg``
    on [1, k_max] rescaled to mean ``k_avg``, a fraction ``mixing`` of every
    node's out-edges leaving its community, targets drawn uniformly inside the
    chosen community.  Memberships are Beta(mean, sd) with the same community
    ``shift`` as ``planted_partition_fuzzy_graph``.  Returns (graph, truth).
    This is a simplified generator (no in-degree power law, no overlap) and
    is described as such in the paper."""
    rng = np.random.default_rng(seed)
    c_max = c_max or max(c_min + 1, n // 10)

    def powerlaw(lo, hi, tau, size):
        u = rng.random(size)
        if abs(tau - 1.0) < 1e-9:
            return lo * (hi / lo) ** u
        a = 1.0 - tau
        return (lo ** a + u * (hi ** a - lo ** a)) ** (1.0 / a)

    sizes = []
    while sum(sizes) < n:
        sizes.append(int(round(powerlaw(c_min, c_max, tau_size, 1)[0])))
    excess = sum(sizes) - n
    if sizes[-1] - excess >= c_min:
        sizes[-1] -= excess
    else:                            # drop the last community and spread the shortfall
        sizes.pop()
        short = n - sum(sizes)
        for i in range(len(sizes)):
            sizes[i] += short // len(sizes) + (1 if i < short % len(sizes) else 0)
    assert sum(sizes) == n
    k = len(sizes)
    truth = np.repeat(np.arange(k), sizes)
    rng.shuffle(truth)
    members = [np.where(truth == c)[0] for c in range(k)]
    deg = powerlaw(1.0, k_max, tau_deg, n)
    deg = deg * (k_avg / deg.mean())
    deg = np.clip(np.round(deg), 1, k_max).astype(int)
    edges = {}
    for u in range(n):
        for _ in range(deg[u]):
            inside = rng.random() >= mixing
            if inside:
                pool = members[truth[u]]
            else:
                c2 = int(rng.integers(0, k))
                while c2 == truth[u] and k > 1:
                    c2 = int(rng.integers(0, k))
                pool = members[c2]
            v = int(rng.choice(pool))
            if v == u or (u, v) in edges:
                continue
            edges[(u, v)] = inside
    e = np.array(sorted(edges))
    inside = np.array([edges[tuple(x)] for x in e])
    d = shift * min(mu_mean, 1 - mu_mean)
    mu = np.empty(len(e))
    a_in, b_in = _beta_params(mu_mean + d, mu_sd)
    a_out, b_out = _beta_params(mu_mean - d, mu_sd)
    mu[inside] = rng.beta(a_in, b_in, size=int(inside.sum()))
    mu[~inside] = rng.beta(a_out, b_out, size=int((~inside).sum()))
    mu = np.clip(mu, mu_min, 1.0)
    return FuzzyGraph.from_edges(zip(e[:, 0], e[:, 1], mu), n=n), truth
