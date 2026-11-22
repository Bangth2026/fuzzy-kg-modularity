"""Grade-preserving baselines (version 4.6.0).

Two models that learn from the memberships instead of thresholding or composing them.  Both
are trained to regress mu_E on the observed edges and 0 on sampled non-edges, and both score
every ordered pair, so neither has a reachability limit.

  ukge   An uncertain-KG embedding in the manner of UKGE (Chen et al., AAAI 2019), logistic
         variant without soft-logic rules: confidence f(u, v) = sigmoid(w * s(u, v) + b) with
         the ComplEx score s(u, v) = Re<e_u, r, conj(e_v)> of one relation "prerequisite of"
         (ComplEx rather than DistMult because a prerequisite graph is directed; on a
         symmetric graph the two coincide in what they can represent).
  gnn    A two-layer GCN encoder on the membership-weighted adjacency (A + A^T, self loops,
         symmetric normalisation; free node embeddings as input features) with an asymmetric
         bilinear decoder f(u, v) = sigmoid(z_u^T R z_v + b).

Training: squared error on edges (target mu_E) and on `neg` non-edges per edge, drawn
uniformly and rejected if they are edges (target 0), with edges and non-edges weighted equally;
Adam; the number of epochs is chosen by
the validation MSE on 10% of the edges held out with their own non-edges, the learning rate
(0.003, 0.01, 0.03) and the dimension (32, 64) by the same validation MSE, and the model is then
refitted on all edges for that number of epochs with three seeds whose predictions are averaged.  Hyperparameters are fixed a priori and are not
tuned on any criterion of the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None


@dataclass
class EmbedConfig:
    model: str = "ukge"          # "ukge" or "gnn"
    dim: int = 64
    lr: float = 0.02
    epochs: int = 200
    eval_every: int = 10
    neg: int = 5
    batch: int = 8192
    weight_decay: float = 0.0
    val_share: float = 0.1
    seed: int = 0


def _require_torch():
    if torch is None:
        raise ImportError("fuzzykg.embeddings needs PyTorch (pip install torch)")


class _UKGE(nn.Module if torch is not None else object):
    def __init__(self, n, d):
        super().__init__()
        self.re = nn.Embedding(n, d); self.im = nn.Embedding(n, d)
        self.r_re = nn.Parameter(torch.randn(d) * 0.1); self.r_im = nn.Parameter(torch.randn(d) * 0.1)
        self.w = nn.Parameter(torch.tensor(1.0)); self.b = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.re.weight, std=0.1); nn.init.normal_(self.im.weight, std=0.1)

    def forward(self, u, v, cache=None):
        hr, hi, tr, ti = self.re(u), self.im(u), self.re(v), self.im(v)
        s = (hr * self.r_re * tr + hi * self.r_re * ti + hr * self.r_im * ti - hi * self.r_im * tr).sum(-1)
        return torch.sigmoid(self.w * s + self.b)

    def encode(self):
        return None


class _GNN(nn.Module if torch is not None else object):
    def __init__(self, n, d, A_hat):
        super().__init__()
        self.x = nn.Embedding(n, d); nn.init.normal_(self.x.weight, std=0.1)
        self.l1 = nn.Linear(d, d); self.l2 = nn.Linear(d, d)
        self.R = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d)); self.b = nn.Parameter(torch.tensor(0.0))
        self.A = A_hat

    def encode(self):
        h = torch.relu(self.l1(torch.sparse.mm(self.A, self.x.weight)))
        return self.l2(torch.sparse.mm(self.A, h))

    def forward(self, u, v, cache=None):
        z = self.encode() if cache is None else cache
        return torch.sigmoid(((z[u] @ self.R) * z[v]).sum(-1) + self.b)


def _a_hat(n, src, dst, mu):
    i = np.r_[src, dst, np.arange(n)]; j = np.r_[dst, src, np.arange(n)]; w = np.r_[mu, mu, np.ones(n)]
    from scipy.sparse import coo_matrix
    A = coo_matrix((w, (i, j)), shape=(n, n)).tocsr(); A.sum_duplicates()
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    A = A.multiply(dinv[:, None]).multiply(dinv[None, :]).tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, values=torch.tensor(A.data, dtype=torch.float32), size=(n, n), check_invariants=False).coalesce()


def _sample_neg(n, k, edge_keys, rng):
    """k uniformly drawn ordered pairs (u, v), u != v, that are not edges (edge_keys sorted)."""
    out_u, out_v, got = [], [], 0
    while got < k:
        u = rng.integers(0, n, 2 * (k - got) + 16); v = rng.integers(0, n, 2 * (k - got) + 16)
        key = u.astype(np.int64) * n + v
        pos = np.searchsorted(edge_keys, key); pos = np.minimum(pos, len(edge_keys) - 1)
        ok = (u != v) & (edge_keys[pos] != key)
        out_u.append(u[ok]); out_v.append(v[ok]); got += int(ok.sum())
    return np.concatenate(out_u)[:k], np.concatenate(out_v)[:k]


def _build(cfg, n, src, dst, mu):
    if cfg.model == "ukge":
        return _UKGE(n, cfg.dim)
    if cfg.model == "gnn":
        return _GNN(n, cfg.dim, _a_hat(n, src, dst, mu))
    raise ValueError(cfg.model)


def _train(cfg, n, src, dst, mu, edge_keys, epochs, rng, val=None, neg_keys=None):
    torch.manual_seed(cfg.seed)
    model = _build(cfg, n, src, dst, mu)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    m = len(src); hist = []
    batch = max(cfg.batch, (1 + cfg.neg) * m // 20)      # at most about 20 optimiser steps per epoch
    for ep in range(1, epochs + 1):
        nu, nv = _sample_neg(n, cfg.neg * m, edge_keys if neg_keys is None else neg_keys, rng)
        U = np.r_[src, nu]; V = np.r_[dst, nv]; Y = np.r_[mu, np.zeros(len(nu))]
        Wt = np.r_[np.full(m, 0.5 / m), np.full(len(nu), 0.5 / len(nu))] * len(U)   # edges and non-edges weigh equally
        perm = rng.permutation(len(U))
        model.train()
        for s in range(0, len(U), batch):
            b = perm[s:s + batch]
            u = torch.as_tensor(U[b]); v = torch.as_tensor(V[b]); y = torch.as_tensor(Y[b], dtype=torch.float32)
            loss = (torch.as_tensor(Wt[b], dtype=torch.float32) * (model(u, v) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if val is not None and ep % cfg.eval_every == 0:
            hist.append((ep, _mse(model, *val)))
    return model, hist


def _mse(model, u, v, y):
    """Validation MSE with edges and non-edges weighted equally, as in training."""
    model.eval()
    with torch.no_grad():
        z = model.encode()
        p = model(torch.as_tensor(u), torch.as_tensor(v), cache=z).numpy()
    e = y > 0
    return float(0.5 * np.mean((p[e] - y[e]) ** 2) + 0.5 * np.mean(p[~e] ** 2))


class GradedLinkModel:
    """fit(G) then score(u, v) -> predicted membership of the ordered pair (u, v)."""

    def __init__(self, cfg: Optional[EmbedConfig] = None, **kw):
        _require_torch()
        self.cfg = cfg or EmbedConfig(**kw)
        self.model = None; self.best_epoch = None; self.val_curve = None

    def fit(self, G, grid=(("lr", (0.003, 0.01, 0.03)), ("dim", (32, 64))), refit_seeds=(0, 1, 2), exclude=None):
        """Select lr and dim by the validation MSE (seed 0), choose the epoch count on the same
        validation split, then refit on all edges with every seed in refit_seeds; score()
        averages the refitted models.  ``exclude``: optional (u, v) pairs never drawn as
        non-edges, in training or validation (version 4.8.0; e.g. the evaluation pairs)."""
        import itertools
        from dataclasses import replace
        torch.set_num_threads(1)
        base = self.cfg
        n, src, dst, mu = G.n, np.asarray(G.src), np.asarray(G.dst), np.asarray(G.mu, float)
        keys = np.sort(src.astype(np.int64) * n + dst)
        nkeys = keys
        if exclude is not None:
            ex = np.asarray(exclude, dtype=np.int64).reshape(-1, 2)
            nkeys = np.unique(np.r_[keys, ex[:, 0] * n + ex[:, 1]])
        rng = np.random.default_rng(base.seed)
        perm = rng.permutation(len(src)); nv = max(1, int(base.val_share * len(src)))
        va, tr = perm[:nv], perm[nv:]
        vu, vv = _sample_neg(n, base.neg * nv, nkeys, rng)
        val = (np.r_[src[va], vu], np.r_[dst[va], vv], np.r_[mu[va], np.zeros(len(vu))])
        names = [g[0] for g in grid]; self.grid_results = []
        best = None
        for combo in itertools.product(*[g[1] for g in grid]):
            cfg = replace(base, **dict(zip(names, combo)))
            _, hist = _train(cfg, n, src[tr], dst[tr], mu[tr], keys, cfg.epochs, np.random.default_rng(cfg.seed), val=val, neg_keys=nkeys)
            ep, mse = min(hist, key=lambda t: t[1])
            self.grid_results.append({**dict(zip(names, combo)), "best_epoch": int(ep), "val_mse": float(mse)})
            if best is None or mse < best[2]:
                best = (cfg, int(ep), float(mse))
        self.cfg, self.best_epoch, self.val_mse = best
        self.models = []
        for sd in refit_seeds:
            cfg = replace(self.cfg, seed=sd)
            m, _ = _train(cfg, n, src, dst, mu, keys, self.best_epoch, np.random.default_rng(1000 + sd), neg_keys=nkeys)
            self.models.append(m)
        self.model = self.models[0]
        return self

    def score_each(self, u, v):
        return [self._score(m, u, v) for m in self.models]

    def score(self, u, v) -> np.ndarray:
        """Mean prediction of the refitted models."""
        return np.mean(self.score_each(u, v), axis=0)

    @staticmethod
    def _score(model, u, v) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            z = model.encode()
            out = []
            u = np.asarray(u); v = np.asarray(v)
            for s in range(0, len(u), 200000):
                out.append(model(torch.as_tensor(u[s:s + 200000]), torch.as_tensor(v[s:s + 200000]), cache=z).numpy())
        return np.concatenate(out) if out else np.zeros(0)
