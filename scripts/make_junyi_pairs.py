#!/usr/bin/env python3
"""Write the Junyi crowd prerequisite scores as an external-pairs file for scripts/fkg_audit.py
(columns pre, post, score: the score that `pre` is a prerequisite of `post`). The annotations are
not redistributed; this reads them from data/raw/junyi/junyi/."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.loaders import load_junyi_expert  # noqa: E402
ex, ann = load_junyi_expert()
out = sys.argv[1] if len(sys.argv) > 1 else "junyi_expert_pairs.csv"
ann.assign(pre=ann.Exercise_B, post=ann.Exercise_A, score=ann.Prerequisite_avg)[["pre", "post", "score"]].to_csv(out, index=False)
print("wrote", out)
