import json, re, sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/kt/src'); import analyse as A
from refs import REFS
df=A.df
D=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet']
N={'assist09':'AS09','assist12':'AS12','assist15':'AS15','assist17':'AS17','junyi':'Junyi','algebra05':'Alg05','eedi':'Eedi','ednet':'EdNet'}
def sel(fr,d,metric):
    g=df[(df.frac==fr)&(df.data==d)&df.cfg.str.startswith('FKG-KT/')&(df.cfg!='FKG-KT/none')].groupby('cfg')
    b=g.valid_auc.mean().idxmax(); return df[(df.frac==fr)&(df.data==d)&(df.cfg==b)][metric]
SD=False
def table2():
    lab=[('DKT','DKT'),('SAKT','SAKT'),('GKT','GKT'),('FKG-KT/none','FKG-KT, no graph'),('FKG-KT/transition:product3','FKG-KT, composed KG (default)'),
         ('FKG-KT/transition:support','FKG-KT, KG support'),('FKG-KT/dktinf:rank','FKG-KT, DKT-influence KG'),('FKG-KT/expert','FKG-KT, expert structure'),('SEL','FKG-KT, selected on validation')]
    out=[]
    for fr,name in ((0.1,'Sparse regime'),(1.0,'Full regime')):
        out.append(f'\n*{name}*\n'); out.append('| Model | '+' | '.join(N[d] for d in D)+' |'); out.append('|:'+'-'*30+'|'+('-'*13+':|')*len(D))
        for c,l in lab:
            if fr==1.0 and c=='FKG-KT/transition:support': continue
            cells=[]; col={}
            for d in D:
                x=(sel(fr,d,'test_auc') if c=='SEL' else df[(df.frac==fr)&(df.data==d)&(df.cfg==c)].test_auc).dropna()
                cells.append('–' if len(x)==0 else (f"{x.mean():.4f}" if (len(x)<2 or not SD) else f"{x.mean():.4f} ({x.std():.4f})"))
            out.append(f'| {l} | '+' | '.join(cells)+' |')
    return '\n'.join(out)
def table3():
    rows=json.load(open('table4.json'))
    L={'transition':'order-preserving','copractice':'co-practice','dktinf':'DKT-influence'}
    out=['| Graph input | AUC diff. | positive | p | first-attempt AUC diff. | positive | p |','|:'+'-'*34+'|'+'-'*10+':|'+'-'*8+':|'+'-'*7+':|'+'-'*12+':|'+'-'*8+':|'+'-'*7+':|']
    for r in rows:
        c=r['cfg'].replace('FKG-KT/','')
        if c=='expert': name='expert structure (4 corpora)'
        else:
            k,g=c.split(':'); name=f"{L[k]}: {g}"
        a=r['test_auc']; b=r['test_auc_newkc']
        f=lambda v:('–' if v is None else f'{v:.3f}')
        out.append(f"| {name} | {a[0]:+.4f} | {a[1]}/{r['n']} | {f(a[2])} | {b[0]:+.4f} | {b[1]}/{r['n']} | {f(b[2])} |")
    return '\n'.join(out)
def table4():
    R=json.load(open('nulltests_all.json'))
    L={'transition':'order-pres.','copractice':'co-practice','dktinf':'DKT-infl.'}
    out=['| Corpus | Construction | Q_F | z degree | z global | z row | grouping ARI (permuted) |','|:'+'-'*8+'|:'+'-'*14+'|'+'-'*8+':|'+'-'*9+':|'+'-'*9+':|'+'-'*8+':|:'+'-'*22+'|']
    for d in D:
        for c in ('transition','copractice','dktinf'):
            v=R[f'{d}/{c}']; g=v.get('grouping_ari')
            gs='–' if g is None else f"{g:.2f} ({v['grouping_ari_permuted'][0]:.2f} ± {v['grouping_ari_permuted'][1]:.2f})"
            out.append(f"| {N[d]} | {L[c]} | {v['Q_F']:.3f} | {v['degree']['z']:.1f} | {v['global']['z']:.1f} | {v['row']['z']:.1f} | {gs} |")
    return '\n'.join(out)
def table1():
    t=pd.read_csv('table1.csv').set_index('data'); exp={'junyi':'prerequisite map','eedi':'subject tree','algebra05':'units','ednet':'TOEIC parts'}
    out=['| Corpus | Learners | Interactions | KCs | Train. int. per KC | Edges: order-pres. / co-practice / DKT-infl. | Expert structure |','|:'+'-'*18+'|'+'-'*9+':|'+'-'*12+':|'+'-'*6+':|'+'-'*10+':|:'+'-'*24+'|:'+'-'*16+'|']
    full={'assist09':'ASSISTments 2009','assist12':'ASSISTments 2012','assist15':'ASSISTments 2015','assist17':'ASSISTments 2017','junyi':'Junyi Academy','algebra05':'Algebra 2005','eedi':'Eedi','ednet':'EdNet KT1'}
    for d in D:
        r=t.loc[d]; out.append(f"| {full[d]} | {int(r.learners):,} | {int(r.interactions):,} | {int(r.KCs)} | {r.train_int_per_kc:.0f} | {int(r.E_trans):,} / {int(r.E_cop):,} / {int(r.E_dkt):,} | {exp.get(d,'–')} |")
    return '\n'.join(out)
front=open('front.md').read()
body=''.join(open(p).read()+'\n\n' for p in ['part0_intro.md','part1_methods.md','part2_setup.md','part3_results.md','part4_discussion.md'])
back=open('back.md').read()
appendix=open('part9_appendix.md').read()
body=body.replace('[TABLE1]',table1()).replace('[TABLE2]',table2()).replace('[TABLE3]',table3()).replace('[TABLE4]',table4())
text=front+body+back
# citations: [KEY, KEY] patterns with known keys
order=[]
def rep(m):
    keys=[k.strip() for k in m.group(1).split(',')]
    if not all(k in REFS for k in keys): return m.group(0)
    for k in keys:
        if k not in order: order.append(k)
    nums=sorted(order.index(k)+1 for k in keys)
    return '['+','.join(map(str,nums))+']'
pat=re.compile(r'\[([A-Za-z][A-Za-z0-9]*(?:, ?[A-Za-z][A-Za-z0-9]*)*)\]')
text=pat.sub(rep,text); appendix=pat.sub(rep,appendix)
unused=[k for k in REFS if k not in order]; print('unused refs',unused)
left=re.findall(r'\[[A-Za-z][A-Za-z0-9]*(?:, ?[A-Za-z0-9]+)*\]',text); print('unresolved',set(left))
refs='\n## References\n\n'+'\n\n'.join(f'[{i+1}] {REFS[k]}' for i,k in enumerate(order))+'\n'
open('manuscript.md','w').write(text+refs+'\n'+appendix+'\n'+open('bio.md').read())
print('words',len((text).split()))
