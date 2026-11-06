"""Memory-light Junyi preprocessing, equivalent to prep.junyi() (same rows, same learner draw).

prep.junyi() loads the 2.6 GB problem log at once, which needs more than 8 GB of RAM. This script
reads it twice in chunks: pass 1 counts interactions per learner, the 6,000 learners are drawn, pass 2
keeps their rows and calls prep.finish, which then draws nothing.

The draw uses prep's generator default_rng(2026) after one earlier 6,000-learner draw, the state in
which release 2.0.0 drew Junyi; it reproduces Table 1 exactly (554,689 interactions after the
200-interaction cap, 691 KCs, correct rate 0.831). Run with --fresh for a fresh generator instead.
Usage: python prep_junyi_lowmem.py [--fresh]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import prep
F = f'{prep.R}/junyi/junyi_ProblemLog_original.csv'
COLS = ['user_id', 'exercise', 'time_done', 'correct']
cnt = None
for c in pd.read_csv(F, usecols=COLS, chunksize=3_000_000):
    s = c.dropna().groupby('user_id').size(); cnt = s if cnt is None else cnt.add(s, fill_value=0)
keep = cnt.sort_index()[lambda x: x >= prep.MINI].index
rng = np.random.default_rng(2026)
if '--fresh' not in sys.argv:
    rng.choice(np.arange(11996), prep.MAXU, replace=False)   # advance by one 6,000-of-N draw (state does not depend on N here)
sel = set(rng.choice(np.array(keep), prep.MAXU, replace=False)) if len(keep) > prep.MAXU else set(keep)
ex = pd.read_csv(f'{prep.R}/junyi/junyi_Exercise_table.csv'); ex.columns = [c.strip('﻿') for c in ex.columns]
topic = dict(zip(ex['name'], ex['topic']))
d = pd.concat([c[c.user_id.isin(sel)] for c in pd.read_csv(F, usecols=COLS, chunksize=3_000_000)])
d['correct'] = d.correct.astype(str).str.lower().eq('true').astype(int)
prep.finish(pd.DataFrame({'user': d.user_id, 'kc': d.exercise, 'correct': d.correct, 'order': d.time_done}), 'junyi', meta=topic)
