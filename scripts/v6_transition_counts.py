#!/usr/bin/env python3
"""Version 4.3.0. Run-to-run move counts T(a -> b) of the Junyi log, for every ordered pair.

The order-preserving construction (Section 4.3) is a function of these counts alone, so the
variants of scripts/v6_transition_sensitivity.py are derived from one counting pass.
Writes data/raw/cache/junyi_transition_counts.parquet (src, dst, T) and the per-source totals.
Usage: python scripts/v6_transition_counts.py
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_any, RAW  # noqa: E402
from fuzzykg.construction import ConstructionConfig, preprocess  # noqa: E402
from fuzzykg.fastcount import encode  # noqa: E402


def counts(d):
    ptr, kc, _, labels = encode(d)
    K = len(labels)
    users = np.repeat(np.arange(ptr.size - 1), np.diff(ptr))
    new_run = np.ones(kc.size, dtype=bool)
    new_run[1:] = (kc[1:] != kc[:-1]) | (users[1:] != users[:-1])
    rk, ru = kc[new_run], users[new_run]
    same = ru[1:] == ru[:-1]
    code = rk[:-1][same].astype(np.int64) * K + rk[1:][same]
    uniq, cnt = np.unique(code, return_counts=True)
    return pd.DataFrame({"src": np.asarray(labels)[uniq // K], "dst": np.asarray(labels)[uniq % K], "T": cnt.astype(np.int64)})


def main():
    out = RAW / "cache" / "junyi_transition_counts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = preprocess(load_any("junyi", "kc"), ConstructionConfig())
    t = counts(d)
    t.to_parquet(out)
    print(len(t), int(t["T"].sum()), out)


if __name__ == "__main__":
    main()
