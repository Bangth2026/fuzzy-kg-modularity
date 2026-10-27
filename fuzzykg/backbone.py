"""Backbone extraction as an alternative to a global threshold.

``disparity_filter`` implements the multiscale backbone of Serrano, Boguñá and
Vespignani (PNAS 2009) on a directed fuzzy graph: an edge (u, v) with
membership mu is tested against the node-local null in which the strength of
u is spread uniformly at random over its k_out(u) out-edges, with p-value
(1 - mu / s_out(u))^(k_out(u) - 1), and likewise at v with the in-strength and
in-degree.  The edge is kept when either p-value is below the significance
level ``alpha_sig``; an edge that is the only out-edge of its source or the
only in-edge of its target is kept by convention, since the test is undefined
at degree one.  The filter is scale-free in the sense of Proposition 3 (it
depends on mu only through the ratios mu / s), but unlike a quantile cut it
keeps a fraction of the edges that depends on how heterogeneous the
memberships around each node are.
"""
from __future__ import annotations

import numpy as np

from .graph import FuzzyGraph


def disparity_pvalues(G: FuzzyGraph):
    """Per-edge p-values at the source (out) and at the target (in)."""
    s_out = G.out_strength(); s_in = G.in_strength()
    k_out = np.bincount(G.src, minlength=G.n); k_in = np.bincount(G.dst, minlength=G.n)
    p_src = (1.0 - G.mu / s_out[G.src]) ** np.maximum(k_out[G.src] - 1, 0)
    p_dst = (1.0 - G.mu / s_in[G.dst]) ** np.maximum(k_in[G.dst] - 1, 0)
    p_src = np.where(k_out[G.src] <= 1, 0.0, p_src)
    p_dst = np.where(k_in[G.dst] <= 1, 0.0, p_dst)
    return p_src, p_dst


def disparity_filter(G: FuzzyGraph, alpha_sig: float = 0.05) -> np.ndarray:
    """Boolean mask of the edges kept by the disparity filter at ``alpha_sig``."""
    p_src, p_dst = disparity_pvalues(G)
    return (p_src < alpha_sig) | (p_dst < alpha_sig)


def backbone(G: FuzzyGraph, alpha_sig: float = 0.05) -> FuzzyGraph:
    """The crisp backbone (unit weights) kept by the disparity filter."""
    keep = disparity_filter(G, alpha_sig)
    if not keep.any():
        raise ValueError("the disparity filter keeps no edge at this level")
    return FuzzyGraph(G.n, G.src[keep], G.dst[keep], np.ones(int(keep.sum())), labels=getattr(G, "labels", None))
