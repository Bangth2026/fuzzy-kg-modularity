"""Loaders that turn the public releases into the (user_id, kc, correct, order)
frame consumed by ``fuzzykg.construction.build_fuzzy_graph``.

Column names follow the public releases; if a mirror renamed a column, pass
``--user/--kc/--item/--correct/--order`` overrides to the build script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"


def _finish(df: pd.DataFrame, user: str, kc: str, correct: str, order: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "user_id": df[user].astype(str),
        "kc": df[kc],
        "correct": pd.to_numeric(df[correct], errors="coerce"),
    })
    out["order"] = pd.to_numeric(df[order], errors="coerce") if order else range(len(df))
    out = out.dropna(subset=["kc", "correct", "order"])
    return out.reset_index(drop=True)


def load_assistments(release: str, level: str = "kc", path: Optional[str] = None, **cols) -> pd.DataFrame:
    """release in {'2009', '2015', '2017'}; level in {'kc', 'item'}."""
    if release == "2009":
        p = Path(path) if path else RAW / "assist2009" / "skill_builder_data.csv"
        df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        if "original" in df.columns:
            df = df[df["original"] == 1]                 # drop scaffolding
        kc = cols.get("kc", "skill_id" if level == "kc" else "problem_id")
        return _finish(df, cols.get("user", "user_id"), kc, cols.get("correct", "correct"),
                       cols.get("order", "order_id"))
    if release == "2015":
        p = Path(path) if path else RAW / "assist2015" / "2015_100_skill_builders_main_problems.csv"
        df = pd.read_csv(p, low_memory=False)
        if level != "kc":
            raise ValueError("ASSISTments 2015 carries no item identifiers; only the KC level is available")
        return _finish(df, cols.get("user", "user_id"), cols.get("kc", "sequence_id"),
                       cols.get("correct", "correct"), cols.get("order", "log_id"))
    if release == "2017":
        p = Path(path) if path else RAW / "assist2017" / "anonymized_full_release_competition_dataset.csv"
        df = pd.read_csv(p, low_memory=False)
        if "original" in df.columns:
            df = df[df["original"] == 1]
        kc = cols.get("kc", "skill" if level == "kc" else "problemId")
        return _finish(df, cols.get("user", "studentId"), kc, cols.get("correct", "correct"),
                       cols.get("order", "startTime"))
    raise ValueError(f"unknown release {release}")


def load_mooccubex_cs(root: Optional[str] = None):
    """MOOCCubeX-CS prerequisite graph (Section 8): intersect the ground-truth
    prerequisite pairs with the assessed concepts and keep edges whose endpoints
    both carry at least one graded problem.  Returns (edges DataFrame with
    src,dst,mu as concept ids, set of assessed concepts)."""
    root = Path(root) if root else RAW / "MOOCCubeX"
    prereq = json.loads((root / "prerequisites" / "cs.json").read_text(encoding="utf-8"))
    assessed = set()
    with open(root / "relations" / "concept-problem.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                assessed.add(parts[0])
    rows = []
    for rec in prereq:
        # the release stores each pair as a dict or a list; accept both shapes
        if isinstance(rec, dict):
            a, b = rec.get("prerequisite") or rec.get("pre") or rec.get("c1"), rec.get("concept") or rec.get("post") or rec.get("c2")
            w = rec.get("confidence", rec.get("score", rec.get("weight", 1.0)))
        else:
            a, b = rec[0], rec[1]
            w = rec[2] if len(rec) > 2 else 1.0
        if a in assessed and b in assessed and a != b:
            rows.append((a, b, float(w)))
    return pd.DataFrame(rows, columns=["src", "dst", "mu"]), assessed
