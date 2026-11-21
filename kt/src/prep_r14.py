"""Release 3.11.0: the two corpora of the confirmatory replication (Section VI-I of the TLT manuscript).

Neither corpus was used to design the probe or in any earlier run. Same rules as prep.py: rows without a KC
dropped, learners with fewer than 15 interactions excluded, at most 6,000 learners drawn with a fresh
default_rng(2026), KCs re-indexed, first 200 interactions kept at load time.

xes3g5m  XES3G5M (Hugging Face Atomi/XES3G5M_interaction_sequences, train and test splits). The release is
         in pyKT form: each learner's log is cut into windows of 200 and every question is expanded into one
         row per KC (is_repeat = 1 for the second and later KCs of a question). Rows with selectmask = 1 and
         is_repeat = 0 are kept, so each answered question contributes one row with its first KC; windows
         are joined in the order of their timestamps. No curriculum unit is released with this copy.
eedi12   Eedi, NeurIPS 2020 Education Challenge tasks 1-2 (Lucy9999/NIPS34: train_data/train_task_1_2.csv
         and metadata/*_task_1_2.csv): a separate release from tasks 3-4 (the corpus 'eedi'), with its own
         questions and learner identifiers, from the same platform and subject tree. KC = the level-3 subject
         of the question, as for 'eedi'; curriculum unit = its level-2 parent. The 6,000 learners are drawn
         before the 1 GB answer metadata is read, so that only their answers are joined.
Usage: python prep_r14.py xes3g5m | eedi12
"""
import sys, os, ast, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import prep
R = prep.R


def xes3g5m():
    d = pd.concat([pd.read_parquet(f'{R}/xes/train.parquet'), pd.read_parquet(f'{R}/xes/test.parquet')])
    rows = []
    for uid, qs, cs, rs, ts, sm, rp in zip(d.uid, d.questions, d.concepts, d.responses, d.timestamps, d.selectmasks, d.is_repeat):
        cs = np.asarray(cs); rs = np.asarray(rs); ts = np.asarray(ts); sm = np.asarray(sm); rp = np.asarray(rp)
        k = (sm == 1) & (rp == 0) & (cs >= 0) & (rs >= 0)
        rows.append(pd.DataFrame({'user': uid, 'kc': cs[k], 'correct': rs[k], 't': ts[k], 'i': np.arange(len(cs))[k]}))
    d = pd.concat(rows).drop_duplicates(['user', 't', 'kc', 'correct'])
    d = d.sort_values(['user', 't', 'i'], kind='stable'); d['order'] = np.arange(len(d))
    return prep.finish(d[['user', 'kc', 'correct', 'order']], 'xes3g5m')


def eedi12():
    t = pd.read_csv(f'{R}/eedi12/train_task_1_2.csv', usecols=['QuestionId', 'UserId', 'AnswerId', 'IsCorrect'])
    q = pd.read_csv(f'{R}/eedi12/question_metadata_task_1_2.csv'); s = pd.read_csv(f'{R}/eedi/metadata/subject_metadata.csv')
    s.columns = [c.strip('﻿') for c in s.columns]
    lvl = dict(zip(s.SubjectId, s.Level)); par = dict(zip(s.SubjectId, s.ParentId))
    def leaf(lst):
        l = ast.literal_eval(lst); l3 = [x for x in l if lvl.get(x) == 3]; return l3[0] if l3 else None
    q['kc'] = q.SubjectId.map(leaf); qk = dict(zip(q.QuestionId, q.kc))
    t['kc'] = t.QuestionId.map(qk); t = t.dropna(subset=['kc'])
    n = t.groupby('UserId').size(); keep = np.array(sorted(n[n >= prep.MINI].index))
    rng = np.random.default_rng(2026)
    sel = set(rng.choice(keep, prep.MAXU, replace=False)) if len(keep) > prep.MAXU else set(keep)
    t = t[t.UserId.isin(sel)]; ids = set(t.AnswerId)
    parts = [c[c.AnswerId.isin(ids)] for c in pd.read_csv(f'{R}/eedi12/answer_metadata_task_1_2.csv', usecols=['AnswerId', 'DateAnswered'], chunksize=2_000_000)]
    t = t.merge(pd.concat(parts), on='AnswerId', how='left').dropna(subset=['DateAnswered'])
    t['order'] = pd.to_datetime(t.DateAnswered).astype('int64') * 1e-9 * 1000 + t.AnswerId % 1000
    grp = {str(int(k)): str(par.get(int(k))) for k in t.kc.unique()}
    t['kc'] = t.kc.astype(int).astype(str)
    return prep.finish(pd.DataFrame({'user': t.UserId, 'kc': t.kc, 'correct': t.IsCorrect, 'order': t.order}), 'eedi12', meta=grp)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
