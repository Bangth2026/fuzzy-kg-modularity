import pandas as pd, numpy as np, sys
sys.path.insert(0,'/home/claude/kt/src'); import prep
R='/home/claude/raw'
parts=[]
for c in pd.read_csv(f'{R}/junyi/junyi_ProblemLog_original.csv',usecols=['user_id','exercise','time_done','correct'],chunksize=3_000_000): parts.append(c)
d=pd.concat(parts); d['correct']=d.correct.astype(str).str.lower().eq('true').astype(int)
df=pd.DataFrame({'user':d.user_id,'kc':d.exercise,'correct':d.correct,'order':d.time_done}).dropna()
df=df.sort_values(['user','order'],kind='stable'); n=df.groupby('user').size(); keep0=n[n>=15].index
print('n_junyi',len(keep0),flush=True)
for name,pre in [('none',[]),('a12',[20056]),('a15',[11996]),('a12a15',[20056,11996])]:
    rng=np.random.default_rng(2026)
    for m in pre: rng.choice(np.arange(m),6000,replace=False)
    keep=rng.choice(np.array(keep0),6000,replace=False)
    sub=df[df.user.isin(set(keep))]
    cap=sub.groupby('user',sort=False).head(200)
    print(name,sub.kc.nunique(),len(cap),flush=True)
