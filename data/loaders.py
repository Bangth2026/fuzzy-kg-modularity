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


def load_junyi(path: Optional[str] = None) -> pd.DataFrame:
    """Junyi Academy problem log (Chang, Hsu and Chen, EDM 2015; release of Jan 2015).
    One row per problem attempt; KC = exercise; correct = first attempt correct
    without hints; order = time_done (microseconds)."""
    p = Path(path) if path else RAW / "junyi" / "junyi" / "junyi_ProblemLog_original.csv"
    df = pd.read_csv(p, usecols=["user_id", "exercise", "time_done", "correct"],
                     dtype={"user_id": "int64", "exercise": "category", "time_done": "int64", "correct": "string"})
    df["correct"] = (df["correct"].str.lower() == "true").astype("int8")
    # version 4.0.0: the exercise column stays categorical (categories in sorted order), which
    # gives the same node coding as strings (fuzzykg.fastcount.encode) at a fraction of the memory
    kc = df["exercise"].cat.remove_unused_categories()
    kc = kc.cat.reorder_categories(sorted(kc.cat.categories.astype(str)))
    out = pd.DataFrame({"user_id": df["user_id"].astype("int64"), "kc": kc,
                        "correct": df["correct"], "order": df["time_done"]})
    out = out.drop_duplicates(subset=["user_id", "order", "kc"], keep="first")
    return out.reset_index(drop=True)


def load_junyi_expert(root: Optional[str] = None):
    """Expert information released with Junyi: the exercise table (knowledge-map
    parent, topic, area) and the crowd-annotated pairwise prerequisite scores
    (1-9, training and testing files pooled).  Returns (exercises, annotations)."""
    root = Path(root) if root else RAW / "junyi" / "junyi"
    ex = pd.read_csv(root / "junyi_Exercise_table.csv", encoding="utf-8-sig")
    ann = pd.concat([pd.read_csv(root / "relationship_annotation_training.csv"),
                     pd.read_csv(root / "relationship_annotation_testing.csv")], ignore_index=True)
    return ex, ann


# ---------------------------------------------------------------------------
# Version 4.0.0: five further public corpora (KC level) and an external grouping of
# their KCs, plus a dispatcher used by the v4 scripts.  Raw files from the USTC EduData
# mirror (base.ustc.edu.cn/data/): ASSISTment/2012-2013-data-with-predictions-4-final.zip,
# KDD_Cup_2010/algebra_2005_2006.zip, KDD_Cup_2010/bridge_to_algebra_2006_2007.zip,
# NIPS2020/public_data.zip (Eedi), EdNet/EdNet-KT1.zip with EdNet/EdNet-Contents.zip.
# ---------------------------------------------------------------------------
import numpy as _np


def load_assist2012(path: Optional[str] = None) -> pd.DataFrame:
    """ASSISTments 2012-13: skill level, original (non-scaffolding) problems, order = problem_log_id."""
    p = Path(path) if path else RAW / "assist2012" / "2012-2013-data-with-predictions-4-final.csv"
    parts = []
    for c in pd.read_csv(p, usecols=["problem_log_id", "user_id", "skill_id", "correct", "original"], chunksize=2_000_000):
        c = c[(c.original == 1)].dropna(subset=["skill_id"])
        c = c[c.correct.isin([0, 1])]
        parts.append(c)
    d = pd.concat(parts)
    d["skill_id"] = d["skill_id"].astype(int)
    return _finish(d, "user_id", "skill_id", "correct", "problem_log_id")


def _kdd(path: Path, kc_col: str):
    d = pd.read_csv(path, sep="\t", usecols=["Row", "Anon Student Id", kc_col, "Correct First Attempt", "Problem Hierarchy"],
                    dtype={"Anon Student Id": "category", kc_col: "string", "Problem Hierarchy": "string"})
    d = d.dropna(subset=[kc_col])
    d["kc"] = d[kc_col].str.split("~~").str[0]
    unit = d.groupby("kc")["Problem Hierarchy"].agg(lambda s: s.str.split(",").str[0].mode().iloc[0]).to_dict()
    out = _finish(d, "Anon Student Id", "kc", "Correct First Attempt", "Row")
    return out, {str(k): str(v) for k, v in unit.items()}


def load_algebra2005(path: Optional[str] = None):
    """KDD Cup 2010, Algebra I 2005-06 (training file): KC = first KC(Default) of a step;
    order = Row; grouping = curriculum unit (first field of the problem hierarchy)."""
    p = Path(path) if path else RAW / "algebra2005" / "algebra_2005_2006" / "algebra_2005_2006_train.txt"
    return _kdd(p, "KC(Default)")


def load_bridge2006(path: Optional[str] = None):
    """KDD Cup 2010, Bridge to Algebra 2006-07 (training file): KC = first KC(SubSkills)."""
    p = Path(path) if path else RAW / "bridge2006" / "bridge_to_algebra_2006_2007" / "bridge_to_algebra_2006_2007_train.txt"
    return _kdd(p, "KC(SubSkills)")


def load_eedi(root: Optional[str] = None):
    """Eedi (NeurIPS 2020 Education Challenge, tasks 3-4): KC = the level-3 subject of a
    question; order = DateAnswered; grouping = the parent (level-2) subject."""
    import ast
    root = Path(root) if root else RAW / "eedi"
    t = pd.read_csv(root / "train_data" / "train_task_3_4.csv")
    a = pd.read_csv(root / "metadata" / "answer_metadata_task_3_4.csv", usecols=["AnswerId", "DateAnswered"])
    q = pd.read_csv(root / "metadata" / "question_metadata_task_3_4.csv")
    s = pd.read_csv(root / "metadata" / "subject_metadata.csv", encoding="utf-8-sig")
    lvl = dict(zip(s.SubjectId, s.Level)); par = dict(zip(s.SubjectId, s.ParentId))

    def leaf(lst):
        l3 = [x for x in ast.literal_eval(lst) if lvl.get(x) == 3]
        return l3[0] if l3 else None
    q["kc"] = q.SubjectId.map(leaf)
    t = t.merge(a, on="AnswerId", how="left")
    t["kc"] = t.QuestionId.map(dict(zip(q.QuestionId, q.kc)))
    t = t.dropna(subset=["kc", "DateAnswered"])
    t["kc"] = t.kc.astype(int).astype(str)
    t["order"] = pd.to_datetime(t.DateAnswered).astype("int64") // 1000 + t.AnswerId % 1000
    grp = {k: str(par.get(int(k))) for k in t.kc.unique()}
    return _finish(t, "UserId", "kc", "IsCorrect", "order"), grp


def load_ednet(n_users: int = 20000, seed: int = 2026, root: Optional[str] = None):
    """EdNet-KT1 (Santa TOEIC): n_users learner files drawn at random; KC = first tag of a
    question; correct = answer equals the key; order = timestamp; grouping = TOEIC part."""
    import zipfile
    root = Path(root) if root else RAW / "ednet"
    q = pd.read_csv(root / "contents" / "questions.csv")
    key = dict(zip(q.question_id, q.correct_answer))
    tag = dict(zip(q.question_id, q.tags.astype(str).str.split(";").str[0]))
    part = dict(zip(q.question_id, q.part))
    z = zipfile.ZipFile(root / "EdNet-KT1.zip")
    names = [n for n in z.namelist() if n.endswith(".csv")]
    pick = _np.random.default_rng(seed).choice(len(names), min(n_users, len(names)), replace=False)
    rows = []
    for i in pick:
        d = pd.read_csv(z.open(names[i]))
        if len(d) < 15:
            continue
        d["user"] = names[i].rsplit("/", 1)[-1][:-4]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["correct"] = (d.user_answer == d.question_id.map(key)).astype(int)
    d["kc"] = d.question_id.map(tag)
    d["order"] = d.timestamp * 1000 + d.solving_id % 1000
    tagpart = {}
    for qid, t in tag.items():
        tagpart.setdefault(str(t), str(part[qid]))
    return _finish(d, "user", "kc", "correct", "order"), tagpart


def load_junyi_with_suggested(path: Optional[str] = None) -> pd.DataFrame:
    p = Path(path) if path else RAW / "junyi" / "junyi" / "junyi_ProblemLog_original.csv"
    df = pd.read_csv(p, usecols=["user_id", "exercise", "time_done", "correct", "suggested"],
                     dtype={"user_id": "int64", "exercise": "category", "time_done": "int64", "correct": "string", "suggested": "string"})
    kc = df["exercise"].cat.remove_unused_categories()
    kc = kc.cat.reorder_categories(sorted(kc.cat.categories.astype(str)))
    out = pd.DataFrame({"user_id": df["user_id"].astype("int64"), "kc": kc,
                        "correct": (df["correct"].str.lower() == "true").astype("int8"), "order": df["time_done"],
                        "suggested": df["suggested"].str.lower() == "true"})
    return out.drop_duplicates(subset=["user_id", "order", "kc"], keep="first").reset_index(drop=True)


NEW_CORPORA = {"assist2012": load_assist2012, "algebra2005": load_algebra2005, "bridge2006": load_bridge2006,
               "eedi": load_eedi, "ednet": load_ednet}


def load_any(release: str, level: str = "kc", keep_suggested: bool = False, use_cache: bool = True):
    """Dispatcher: '2009', '2015', '2017', 'junyi', or one of NEW_CORPORA (KC level)."""
    if use_cache and release in NEW_CORPORA:
        return cached(release, level)[0]
    if release == "junyi":
        return load_junyi_with_suggested() if keep_suggested else load_junyi()
    if release in ("2009", "2015", "2017"):
        return load_assistments(release, level)
    r = NEW_CORPORA[release]()
    return r[0] if isinstance(r, tuple) else r


def load_groups(release: str):
    r = NEW_CORPORA[release]()
    return r[1] if isinstance(r, tuple) else None


def cached(release: str, level: str = "kc", keep_suggested: bool = False):
    """load_any with a parquet cache under data/raw/cache (and the grouping as JSON)."""
    c = RAW / "cache"; c.mkdir(parents=True, exist_ok=True)
    tag = f"{release}_{level}" + ("_sugg" if keep_suggested else "")
    f = c / f"{tag}.parquet"; g = c / f"{tag}_groups.json"
    if f.exists():
        return pd.read_parquet(f), (json.load(open(g)) if g.exists() else None)
    if release in NEW_CORPORA:
        r = NEW_CORPORA[release]()
        df, grp = (r if isinstance(r, tuple) else (r, None))
    else:
        df, grp = load_any(release, level, keep_suggested), None
    df["user_id"] = df["user_id"].astype(str); df["kc"] = df["kc"].astype(str)
    df.to_parquet(f)
    if grp is not None:
        json.dump(grp, open(g, "w"))
    return df, grp
