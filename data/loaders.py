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
    # one log entry per (learner, log order): the 2009 release repeats a row
    # once per skill of a multi-skill item; the first is kept (Table 3 counts)
    out = out.drop_duplicates(subset=["user_id", "order"], keep="first")
    return out.reset_index(drop=True)


def load_assistments(release: str, level: str = "kc", path: Optional[str] = None, **cols) -> pd.DataFrame:
    """release in {'2009', '2015', '2017'}; level in {'kc', 'item'}."""
    if release == "2009":
        p = Path(path) if path else RAW / "assist2009" / "skill_builder_data.csv"
        df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        if "original" in df.columns:
            df = df[df["original"] == 1]                 # drop scaffolding
        df = df.dropna(subset=["skill_id"])              # rows with missing skill are dropped at both levels
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
        df = df.dropna(subset=["skill"])
        kc = cols.get("kc", "skill" if level == "kc" else "problemId")
        return _finish(df, cols.get("user", "studentId"), kc, cols.get("correct", "correct"),
                       cols.get("order", "startTime"))
    raise ValueError(f"unknown release {release}")


def load_mooccubex_cs(root: Optional[str] = None, discipline: str = "计算机科学与技术", full: bool = False):
    """MOOCCubeX-CS prerequisite graph (Section 8): intersect the ground-truth
    prerequisite pairs with the assessed concepts and keep edges whose endpoints
    both carry at least one graded problem.  With ``full=True`` the assessed
    restriction is dropped and all ground-truth pairs are kept (the full
    expert-annotated CS graph).  Returns (edges DataFrame with src,dst,mu as
    concept ids, set of assessed concepts)."""
    root = Path(root) if root else RAW / "MOOCCubeX"
    # prerequisites/cs.json is JSON-lines: one record per candidate pair with
    # fields c1, c2, ground_truth and two predicted confidences; the paper keeps
    # the 725 ground-truth pairs and reads mu_E from graph_predict[1].
    # relations/concept-problem.txt has ids "K_<name>_<discipline>\tPm_<id>";
    # the CS discipline (计算机科学与技术, 1857 concepts) is kept, matched by name.
    prereq = [json.loads(l) for l in (root / "prerequisites" / "cs.json").read_text(encoding="utf-8").splitlines() if l.strip()]
    assessed = set()
    with open(root / "relations" / "concept-problem.txt", encoding="utf-8") as f:
        for line in f:
            cid = line.rstrip("\n").split("\t")[0]
            if cid.startswith("K_"):
                name, field = cid[2:].rsplit("_", 1)
                if field == discipline:
                    assessed.add(name)
    rows = []
    for rec in prereq:
        if int(rec.get("ground_truth", 0)) != 1:
            continue
        a, b = rec["c1"], rec["c2"]
        w = rec["graph_predict"][1]
        if (full or (a in assessed and b in assessed)) and a != b:
            rows.append((a, b, float(w)))
    return pd.DataFrame(rows, columns=["src", "dst", "mu"]), assessed
