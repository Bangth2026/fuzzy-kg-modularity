"""Compiled counting kernels for the graph constructions (Section 7.1).

The pure-Python loops in ``construction.py`` are exact but slow on logs with
tens of millions of interactions (Junyi Academy).  The kernels below compute
the same quantities on integer-coded arrays with numba; ``construction.py``
uses them when numba is installed, and ``tests/test_fastcount.py`` checks that
both paths give identical graphs on a small log.

All kernels take learners as contiguous blocks: ``ptr[u]:ptr[u+1]`` indexes the
interactions of learner ``u`` in log order, ``kc`` holds integer KC codes in
``0..K-1`` and ``correct`` holds 0/1.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # pragma: no cover - numba is optional
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def precedence_matrix(ptr, kc, K, horizon):
    """N[a, b] = number of (t, t') of the same learner with t < t' <= t + horizon,
    kc_t = a, kc_t' = b, a != b."""
    N = np.zeros((K, K), dtype=np.int64)
    for u in range(ptr.size - 1):
        s, e = ptr[u], ptr[u + 1]
        for i in range(s, e):
            a = kc[i]
            stop = min(e, i + horizon + 1)
            for j in range(i + 1, stop):
                b = kc[j]
                if a != b:
                    N[a, b] += 1
    return N


@njit(cache=True)
def mastery_counts(ptr, kc, correct, K, wanted_ptr, wanted_a, threshold):
    """For every wanted pair (a, b): sums and counts of correctness on b over the
    attempts made when a was already partially mastered (running accuracy on a
    reached ``threshold`` at some earlier point) and over the attempts made before
    any attempt on a.  ``wanted_ptr``/``wanted_a`` list, for each b, the a's with
    (a, b) wanted (CSR layout)."""
    wc = np.zeros((K, K), dtype=np.int64)
    wn = np.zeros((K, K), dtype=np.int64)
    oc = np.zeros((K, K), dtype=np.int64)
    on = np.zeros((K, K), dtype=np.int64)
    seen_n = np.zeros(K, dtype=np.int64)
    seen_c = np.zeros(K, dtype=np.int64)
    mastered = np.zeros(K, dtype=np.bool_)
    touched = np.empty(K, dtype=np.int64)
    for u in range(ptr.size - 1):
        s, e = ptr[u], ptr[u + 1]
        nt = 0
        for i in range(s, e):
            b = kc[i]
            c = correct[i]
            for q in range(wanted_ptr[b], wanted_ptr[b + 1]):
                a = wanted_a[q]
                if mastered[a]:
                    wc[a, b] += c
                    wn[a, b] += 1
                elif seen_n[a] == 0:
                    oc[a, b] += c
                    on[a, b] += 1
            if seen_n[b] == 0:
                touched[nt] = b
                nt += 1
            seen_n[b] += 1
            seen_c[b] += c
            if seen_c[b] >= threshold * seen_n[b]:
                mastered[b] = True
        for t in range(nt):          # reset only what this learner touched
            k = touched[t]
            seen_n[k] = 0
            seen_c[k] = 0
            mastered[k] = False
    return wc, wn, oc, on


@njit(cache=True)
def transition_matrix(ptr, kc, K):
    """T[a, b] = number of moves from KC a to a different KC b between consecutive
    runs of the same learner (repeats of one KC are collapsed into a run)."""
    T = np.zeros((K, K), dtype=np.int64)
    for u in range(ptr.size - 1):
        s, e = ptr[u], ptr[u + 1]
        prev = -1
        for i in range(s, e):
            k = kc[i]
            if k != prev:
                if prev >= 0:
                    T[prev, k] += 1
                prev = k
    return T


def encode(d):
    """Integer-code a preprocessed frame (sorted by learner and log order).
    Returns (ptr, kc codes, correct, labels)."""
    labels, kc = np.unique(d["kc"].astype(str).to_numpy(), return_inverse=True)
    users = d["user_id"].to_numpy()
    change = np.ones(len(users), dtype=bool)
    change[1:] = users[1:] != users[:-1]
    starts = np.flatnonzero(change)
    ptr = np.append(starts, len(users)).astype(np.int64)
    return ptr, kc.astype(np.int64), d["correct"].to_numpy().astype(np.int64), labels
