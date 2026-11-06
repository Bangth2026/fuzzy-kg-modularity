"""Unified preprocessing: user, kc, correct, order for 8 corpora (KC-level)."""
import pandas as pd, numpy as np, zipfile, io, json, sys, ast
R='/home/claude/raw'; O='/home/claude/kt/data'
MAXU=6000; MINI=15; RNG=np.random.default_rng(2026)
def finish(df,name,meta=None):
    df=df.dropna(subset=['user','kc','correct','order'])
    df['correct']=df['correct'].astype(int).clip(0,1)
    df=df.sort_values(['user','order'],kind='stable')
    n=df.groupby('user').size(); keep=n[n>=MINI].index
    if len(keep)>MAXU: keep=RNG.choice(np.array(keep),MAXU,replace=False)
    df=df[df.user.isin(set(keep))].copy()
    kcs=sorted(df.kc.astype(str).unique()); kmap={k:i for i,k in enumerate(kcs)}
    df['kc_name']=df.kc.astype(str); df['kc']=df.kc_name.map(kmap)
    df['user']=df.user.astype(str)
    df=df[['user','kc','kc_name','correct','order']].reset_index(drop=True)
    df.to_parquet(f'{O}/{name}.parquet')
    m={'name':name,'interactions':len(df),'learners':df.user.nunique(),'kcs':len(kcs),'correct_rate':float(df.correct.mean())}
    if meta is not None:
        json.dump({k:meta.get(k) for k in kcs},open(f'{O}/{name}_groups.json','w'))
    print(m); return m
def a09():
    d=pd.read_csv(f'{R}/a09/skill_builder_data_corrected.csv',encoding='latin-1',low_memory=False,usecols=['order_id','user_id','skill_id','correct','original'])
    d=d[d.original==1].dropna(subset=['skill_id']).drop_duplicates(['user_id','order_id'])
    return finish(pd.DataFrame({'user':d.user_id,'kc':d.skill_id.astype(int),'correct':d.correct,'order':d.order_id}),'assist09')
def a12():
    parts=[]
    for c in pd.read_csv(f'{R}/a12/2012-2013-data-with-predictions-4-final.csv',usecols=['problem_log_id','user_id','skill_id','correct','original'],chunksize=2_000_000):
        c=c[(c.original==1)].dropna(subset=['skill_id']); c=c[c.correct.isin([0,1])]; parts.append(c)
    d=pd.concat(parts)
    return finish(pd.DataFrame({'user':d.user_id,'kc':d.skill_id.astype(int),'correct':d.correct,'order':d.problem_log_id}),'assist12')
def a15():
    d=pd.read_csv(f'{R}/a15/2015_100_skill_builders_main_problems.csv'); d=d[d.correct.isin([0,1])]
    return finish(pd.DataFrame({'user':d.user_id,'kc':d.sequence_id,'correct':d.correct,'order':d.log_id}),'assist15')
def a17():
    d=pd.read_csv(f'{R}/a17/anonymized_full_release_competition_dataset.csv',usecols=['studentId','skill','correct','original','startTime','action_num'],low_memory=False)
    d=d[d.original==1].dropna(subset=['skill'])
    return finish(pd.DataFrame({'user':d.studentId,'kc':d.skill,'correct':d.correct,'order':d.startTime*1e4+d.action_num%10000}),'assist17')
def junyi():
    ex=pd.read_csv(f'{R}/junyi/junyi_Exercise_table.csv'); ex.columns=[c.strip('﻿') for c in ex.columns]
    topic=dict(zip(ex['name'],ex['topic']))
    parts=[]
    for c in pd.read_csv(f'{R}/junyi/junyi_ProblemLog_original.csv',usecols=['user_id','exercise','time_done','correct'],chunksize=3_000_000):
        parts.append(c)
    d=pd.concat(parts)
    d['correct']=d.correct.astype(str).str.lower().eq('true').astype(int)
    return finish(pd.DataFrame({'user':d.user_id,'kc':d.exercise,'correct':d.correct,'order':d.time_done}),'junyi',meta=topic)
def alg05():
    d=pd.read_csv(f'{R}/alg05/algebra_2005_2006/algebra_2005_2006_train.txt',sep='\t',usecols=['Row','Anon Student Id','KC(Default)','Correct First Attempt','Problem Hierarchy'])
    d=d.dropna(subset=['KC(Default)']); d['kc']=d['KC(Default)'].str.split('~~').str[0]
    unit=d.groupby('kc')['Problem Hierarchy'].agg(lambda s:s.str.split(',').str[0].mode().iloc[0]).to_dict()
    return finish(pd.DataFrame({'user':d['Anon Student Id'],'kc':d.kc,'correct':d['Correct First Attempt'],'order':d.Row}),'algebra05',meta=unit)
def eedi():
    t=pd.read_csv(f'{R}/eedi/train_data/train_task_3_4.csv'); a=pd.read_csv(f'{R}/eedi/metadata/answer_metadata_task_3_4.csv',usecols=['AnswerId','DateAnswered'])
    q=pd.read_csv(f'{R}/eedi/metadata/question_metadata_task_3_4.csv'); s=pd.read_csv(f'{R}/eedi/metadata/subject_metadata.csv'); s.columns=[c.strip('﻿') for c in s.columns]
    lvl=dict(zip(s.SubjectId,s.Level)); par=dict(zip(s.SubjectId,s.ParentId))
    def leaf(lst):
        l=ast.literal_eval(lst); l3=[x for x in l if lvl.get(x)==3]; return l3[0] if l3 else None
    q['kc']=q.SubjectId.map(leaf); qk=dict(zip(q.QuestionId,q.kc))
    t=t.merge(a,on='AnswerId',how='left'); t['kc']=t.QuestionId.map(qk); t=t.dropna(subset=['kc','DateAnswered'])
    t['order']=pd.to_datetime(t.DateAnswered).astype('int64')*1e-9*1000+t.AnswerId%1000
    grp={str(int(k)):str(par.get(int(k))) for k in t.kc.unique()}
    t['kc']=t.kc.astype(int).astype(str)
    return finish(pd.DataFrame({'user':t.UserId,'kc':t.kc,'correct':t.IsCorrect,'order':t.order}),'eedi',meta=grp)
def ednet():
    q=pd.read_csv(f'{R}/ednet/contents/questions.csv'); ans=dict(zip(q.question_id,q.correct_answer)); tag=dict(zip(q.question_id,q.tags.astype(str).str.split(';').str[0])); part=dict(zip(q.question_id,q.part))
    z=zipfile.ZipFile(f'{R}/EdNet-KT1.zip'); names=[n for n in z.namelist() if n.endswith('.csv')]
    pick=RNG.choice(len(names),20000,replace=False); rows=[]
    for i in pick:
        d=pd.read_csv(z.open(names[i]));
        if len(d)<MINI: continue
        d['user']=names[i][4:-4]; rows.append(d)
    d=pd.concat(rows)
    d['correct']=(d.user_answer==d.question_id.map(ans)).astype(int); d['kc']=d.question_id.map(tag)
    tagpart={}
    for qid,t in tag.items(): tagpart.setdefault(t,part[qid])
    return finish(pd.DataFrame({'user':d.user,'kc':d.kc,'correct':d.correct,'order':d.timestamp*1000+d.solving_id%1000}),'ednet',meta={k:str(v) for k,v in tagpart.items()})

def bridge06():
    d=pd.read_csv(f'{R}/bridge/bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_train.txt',sep='\t',usecols=['Row','Anon Student Id','KC(SubSkills)','Correct First Attempt','Problem Hierarchy'])
    d=d.dropna(subset=['KC(SubSkills)']); d['kc']=d['KC(SubSkills)'].str.split('~~').str[0]
    unit=d.groupby('kc')['Problem Hierarchy'].agg(lambda s:s.str.split(',').str[0].mode().iloc[0]).to_dict()
    return finish(pd.DataFrame({'user':d['Anon Student Id'],'kc':d.kc,'correct':d['Correct First Attempt'],'order':d.Row}),'bridge06',meta=unit)
def statics():
    d=pd.read_csv(f'{R}/statics/AllData_student_step_2011F.csv',usecols=['Row','Anon Student Id','KC (F2011)','First Attempt','Problem Hierarchy'],encoding='utf-8-sig')
    d=d.dropna(subset=['KC (F2011)']); d=d[d['First Attempt'].isin(['correct','incorrect','hint'])]
    d['kc']=d['KC (F2011)'].str.split('~~').str[0]; d['c']=(d['First Attempt']=='correct').astype(int)
    unit=d.groupby('kc')['Problem Hierarchy'].agg(lambda s:s.astype(str).str.split(',').str[1].mode().iloc[0]).to_dict()
    return finish(pd.DataFrame({'user':d['Anon Student Id'],'kc':d.kc,'correct':d.c,'order':d.Row}),'statics11',meta=unit)
if __name__=='__main__':
    out={}
    for f in sys.argv[1:]: out[f]=globals()[f]()
    json.dump(out,open(f'{O}/stats_{"_".join(sys.argv[1:])}.json','w'),indent=1)
