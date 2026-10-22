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
