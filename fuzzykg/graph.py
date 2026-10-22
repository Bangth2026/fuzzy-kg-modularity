"""Directed fuzzy graph container (Section 2.1, Section 3.1).

A fuzzy knowledge graph is G_F = (C, E, mu_E) with mu_E : E -> (0,1].
Nodes are integers 0..n-1; an optional ``labels`` list maps them to names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class FuzzyGraph:
    n: int
    src: np.ndarray            # int array, shape (m,)
    dst: np.ndarray            # int array, shape (m,)
    mu: np.ndarray             # float array in (0,1], shape (m,)
    labels: Optional[List[str]] = None
    _adj: Optional[sp.csr_matrix] = field(default=None, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[int, int, float]],
        n: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
        prune_below: float = 0.0,
    ) -> "FuzzyGraph":
        e = [(int(u), int(v), float(w)) for u, v, w in edges if float(w) > prune_below and u != v]
        if not e:
            raise ValueError("no edges after pruning")
        src = np.array([x[0] for x in e], dtype=np.int64)
        dst = np.array([x[1] for x in e], dtype=np.int64)
        mu = np.array([x[2] for x in e], dtype=float)
        if mu.min() <= 0 or mu.max() > 1:
            raise ValueError("memberships must lie in (0,1]")
        if n is None:
            n = int(max(src.max(), dst.max())) + 1
        return cls(n=n, src=src, dst=dst, mu=mu, labels=list(labels) if labels is not None else None)

    @classmethod
    def from_adjacency(cls, A: np.ndarray | sp.spmatrix, labels=None) -> "FuzzyGraph":
        A = sp.coo_matrix(A)
        mask = (A.data > 0) & (A.row != A.col)
        return cls.from_edges(zip(A.row[mask], A.col[mask], A.data[mask]), n=A.shape[0], labels=labels)

    # ------------------------------------------------------------- accessors
    @property
    def m(self) -> int:
        return int(self.mu.size)

    @property
    def W(self) -> float:
        """Total fuzzy mass W = sum of all memberships (Eq. 7)."""
        return float(self.mu.sum())

    def adjacency(self) -> sp.csr_matrix:
        if self._adj is None:
            self._adj = sp.csr_matrix((self.mu, (self.src, self.dst)), shape=(self.n, self.n))
        return self._adj

    def out_strength(self) -> np.ndarray:
        return np.asarray(self.adjacency().sum(axis=1)).ravel()

    def in_strength(self) -> np.ndarray:
        return np.asarray(self.adjacency().sum(axis=0)).ravel()

    def alpha_cut(self, alpha: float) -> "FuzzyGraph":
        """Crisp alpha-cut E_alpha = {e : mu_E(e) >= alpha} with unit weights."""
        keep = self.mu >= alpha
        if not keep.any():
            raise ValueError(f"alpha-cut at {alpha} is empty")
        return FuzzyGraph(self.n, self.src[keep], self.dst[keep], np.ones(int(keep.sum())), self.labels)

    def symmetrised(self) -> sp.csr_matrix:
        A = self.adjacency()
        return (A + A.T).tocsr()

    def reciprocity(self) -> float:
        """Fraction of directed edges whose reverse is also present."""
        A = self.adjacency()
        both = A.multiply(A.T)
        return float(both.nnz) / float(A.nnz)

    def to_igraph(self):
        import igraph as ig

        g = ig.Graph(n=self.n, edges=list(zip(self.src.tolist(), self.dst.tolist())), directed=True)
        g.es["weight"] = self.mu.tolist()
        return g

    def summary(self) -> str:
        return (
            f"FuzzyGraph(n={self.n}, m={self.m}, W={self.W:.3f}, "
            f"mu min/mean/max={self.mu.min():.3f}/{self.mu.mean():.3f}/{self.mu.max():.3f}, "
            f"reciprocity={self.reciprocity():.3f})"
        )
