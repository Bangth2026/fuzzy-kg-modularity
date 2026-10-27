"""Fuzzification of a point estimate with epistemic uncertainty (Section 5).

Eq. (10)  lo = p* - eta*sigma, hi = p* + eta*sigma
Eq. (11)  mu~_unk = clip[(a_par - lo)/(a_par - a_unk)], mu~_mas = clip[(hi - a_par)/(a_mas - a_par)],
          mu~_par = max(0, 1 - mu~_unk - mu~_mas)
Eq. (12)  normalisation onto the 2-simplex  (the "type-1 collapse")
Eq. (13)  footprint of uncertainty: lower/upper membership of each label over [lo, hi]
Section 5.3  h* = (1 - tau_ZPD) * max(a, b) / 2, the half-width beyond which the
          collapse rejects every candidate.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

DEFAULT_CUTS = (0.35, 0.60, 0.80)
LABELS = ("unk", "par", "mas")


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


# ----------------------------------------------------------------- type-1 partition
def partition_memberships(x, cuts=DEFAULT_CUTS):
    """Trapezoidal (Ruspini) partition m_unk, m_par, m_mas evaluated at x (eta = 0)."""
    a_unk, a_par, a_mas = cuts
    x = np.asarray(x, dtype=float)
    m_unk = _clip01((a_par - x) / (a_par - a_unk))
    m_mas = _clip01((x - a_par) / (a_mas - a_par))
    m_par = np.maximum(0.0, 1.0 - m_unk - m_mas)
    return m_unk, m_par, m_mas


# ----------------------------------------------------------------- type-1 collapse
def collapse_type1(p, halfwidth, cuts=DEFAULT_CUTS, eps: float = 1e-9):
    """Eq. (11)-(12).  ``halfwidth`` is eta*sigma.  Returns (mu_unk, mu_par, mu_mas)
    normalised onto the simplex."""
    a_unk, a_par, a_mas = cuts
    p = np.asarray(p, dtype=float)
    h = np.asarray(halfwidth, dtype=float)
    lo, hi = p - h, p + h
    mu_unk = _clip01((a_par - lo) / (a_par - a_unk))
    mu_mas = _clip01((hi - a_par) / (a_mas - a_par))
    mu_par = np.maximum(0.0, 1.0 - mu_unk - mu_mas)
    s = np.maximum(eps, mu_unk + mu_par + mu_mas)
    return mu_unk / s, mu_par / s, mu_mas / s


# ----------------------------------------------------------- footprint of uncertainty
def footprint(p, halfwidth, cuts=DEFAULT_CUTS) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Eq. (13): closed-form lower/upper membership of each label over [lo, hi].

    Uses only shape properties: m_unk non-increasing, m_mas non-decreasing,
    m_par unimodal with mode at a_par.
    """
    a_unk, a_par, a_mas = cuts
    p = np.asarray(p, dtype=float)
    h = np.asarray(halfwidth, dtype=float)
    lo, hi = p - h, p + h
    u_lo, r_lo, m_lo = partition_memberships(lo, cuts)
    u_hi, r_hi, m_hi = partition_memberships(hi, cuts)
    unk = (u_hi, u_lo)                      # non-increasing: min at hi, max at lo
    mas = (m_lo, m_hi)                      # non-decreasing: min at lo, max at hi
    par_lower = np.minimum(r_lo, r_hi)
    inside = (lo <= a_par) & (a_par <= hi)
    par_upper = np.where(inside, 1.0, np.maximum(r_lo, r_hi))
    return {"unk": unk, "par": (par_lower, par_upper), "mas": mas}


def nie_tan(fou: Dict[str, Tuple[np.ndarray, np.ndarray]], renormalise: bool = False):
    """Nie-Tan type reduction: pointwise average of the two bounds, one pass."""
    nt = {k: 0.5 * (lo + hi) for k, (lo, hi) in fou.items()}
    if renormalise:
        s = nt["unk"] + nt["par"] + nt["mas"]
        nt = {k: v / np.maximum(s, 1e-9) for k, v in nt.items()}
    return nt


# ------------------------------------------------------------------------- gates
def zpd_gate_type1(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS):
    """(P1) on the collapsed triple: pass iff mu_par >= tau_ZPD."""
    _, mu_par, _ = collapse_type1(p, halfwidth, cuts)
    return mu_par >= tau_zpd


def zpd_gate_it2(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS):
    """Three-valued gate on the footprint bounds (Section 5.2).

    Returns an int array: 1 = pass (lower >= tau), 0 = fail (upper < tau),
    -1 = indeterminate (lower < tau <= upper).
    """
    lo, hi = footprint(p, halfwidth, cuts)["par"]
    out = np.full(np.broadcast(lo, hi).shape, -1, dtype=int)
    out[lo >= tau_zpd] = 1
    out[hi < tau_zpd] = 0
    return out


def zpd_gate_it2_normalised(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS, eps: float = 1e-9):
    """Variant kept for provenance (see README, note 1): the lower-bound vector
    and the upper-bound vector are each normalised onto the simplex *before*
    gating.  Separate normalisation does not preserve lower <= upper, so the
    pass test (normalised lower >= tau) and the fail test (normalised upper <
    tau) can both fire at the same state; both masks are returned so the
    overlap can be inspected.  See README, "Known discrepancies".
    """
    fou = footprint(p, halfwidth, cuts)
    L = np.stack([fou[k][0] for k in LABELS]); U = np.stack([fou[k][1] for k in LABELS])
    Ln = L / np.maximum(eps, L.sum(axis=0)); Un = U / np.maximum(eps, U.sum(axis=0))
    passed = Ln[1] >= tau_zpd
    failed = Un[1] < tau_zpd
    indeterminate = ~passed & ~failed
    return passed, failed, indeterminate


def embedded_in_footprint(p, halfwidth, cuts=DEFAULT_CUTS, tol: float = 1e-12):
    """For each label, whether the collapsed value lies within [lower, upper]."""
    fou = footprint(p, halfwidth, cuts)
    tri = dict(zip(LABELS, collapse_type1(p, halfwidth, cuts)))
    return {k: (fou[k][0] - tol <= tri[k]) & (tri[k] <= fou[k][1] + tol) for k in LABELS}


# ------------------------------------------------------------ closed-form threshold
def degeneration_halfwidth(tau_zpd: float = 0.4, cuts=DEFAULT_CUTS) -> float:
    """h* = (1 - tau_ZPD) * max(a, b) / 2 with a = a_par - a_unk, b = a_mas - a_par."""
    a_unk, a_par, a_mas = cuts
    a, b = a_par - a_unk, a_mas - a_par
    return 0.5 * (1.0 - tau_zpd) * max(a, b)


def max_collapsed_par(halfwidth: float, cuts=DEFAULT_CUTS) -> float:
    """max over p* of the collapsed mu_par = 1 - 2h / max(a, b)  (Section 5.3)."""
    a_unk, a_par, a_mas = cuts
    a, b = a_par - a_unk, a_mas - a_par
    return 1.0 - 2.0 * halfwidth / max(a, b)


# ------------------------------------------- alternative gates (Supplementary Table S5)
def gate_point_estimate(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS):
    """Baseline (i): gate on m_par(p) only, ignoring the dispersion."""
    _, m_par, _ = partition_memberships(p, cuts)
    return np.broadcast_to(m_par >= tau_zpd, np.broadcast(p, halfwidth).shape)


def gate_interval_image_type1(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS):
    """Baseline (ii): type-1 gate on the interval image, pass iff
    min_{x in [lo,hi]} m_par(x) >= tau.  This is the 'pass' rule of the IT2 gate
    and needs no type-2 machinery."""
    lo_par, _ = footprint(p, halfwidth, cuts)["par"]
    return lo_par >= tau_zpd


def prob_in_zpd_band(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS, eta: float = 1.0):
    """Baseline (iii): P(m_par(X) >= tau) under X ~ N(p, sigma^2), sigma = halfwidth/eta.
    The set {x : m_par(x) >= tau} is the interval [a_par - (1-tau) a, a_par + (1-tau) b]."""
    from scipy.stats import norm
    a_unk, a_par, a_mas = cuts
    a, b = a_par - a_unk, a_mas - a_par
    lo_x, hi_x = a_par - (1 - tau_zpd) * a, a_par + (1 - tau_zpd) * b
    p = np.asarray(p, float); h = np.asarray(halfwidth, float)
    sigma = np.maximum(h / eta, 1e-12)
    return norm.cdf((hi_x - p) / sigma) - norm.cdf((lo_x - p) / sigma)


def gate_probability(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS, p_min: float = 0.5, eta: float = 1.0):
    """Baseline (iii) thresholded: pass iff P(m_par(X) >= tau) >= p_min."""
    return prob_in_zpd_band(p, halfwidth, tau_zpd, cuts, eta) >= p_min


def expected_par_membership(p, halfwidth, cuts=DEFAULT_CUTS, n: int = 65):
    """Baseline (iv): mean of m_par over the interval [lo, hi] (uniform weight)."""
    p = np.asarray(p, float); h = np.asarray(halfwidth, float)
    t = np.linspace(-1.0, 1.0, n)
    xs = p[..., None] + h[..., None] * t
    _, m_par, _ = partition_memberships(xs, cuts)
    return m_par.mean(axis=-1)


def gate_expected_membership(p, halfwidth, tau_zpd: float = 0.4, cuts=DEFAULT_CUTS):
    return expected_par_membership(p, halfwidth, cuts) >= tau_zpd
