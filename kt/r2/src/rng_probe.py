import pandas as pd, numpy as np, sys
R='/home/claude/raw'
def n_users(df,u='user'):
    n=df.groupby(u).size(); return int((n>=15).sum())
a15=pd.read_csv(f'{R}/a15/2015_100_skill_builders_main_problems.csv'); a15=a15[a15.correct.isin([0,1])]
print('a15',n_users(a15,'user_id'),flush=True)
parts=[]
for c in pd.read_csv(f'{R}/a12/2012-2013-data-with-predictions-4-final.csv',usecols=['problem_log_id','user_id','skill_id','correct','original'],chunksize=2_000_000):
    c=c[(c.original==1)].dropna(subset=['skill_id']); c=c[c.correct.isin([0,1])]; parts.append(c[['user_id']])
print('a12',n_users(pd.concat(parts),'user_id'),flush=True)
