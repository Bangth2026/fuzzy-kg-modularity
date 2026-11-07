"""Single source of truth for the numbers of the learning-curve runs (run set R2).
Builds a flat dictionary S of formatted values from paper/r2_results.json and paper/r2_runs.csv.
render(template) fills {key} fields and raises KeyError on any key that is not computed here."""
import json, string, numpy as np, pandas as pd
K='/home/claude/kt'
o=json.load(open(f'{K}/paper/r2_results.json')); runs=pd.read_csv(f'{K}/paper/r2_runs.csv')
D10=['assist09','assist12','assist15','assist17','junyi','algebra05','eedi','ednet','bridge06','statics11']
NAMES=['AS09','AS12','AS15','AS17','Junyi','Alg05','Eedi','EdNet','Bridge06','Statics11']
S={}
def f3(x): return f'{x:.3f}'
def f4(x): return f'{x:+.4f}'.replace('+','+')
def sg(x,d=4): return (f'{x:+.{d}f}').replace('-','−')
def pv(p):
    if p<0.001: return '< 0.001'
    return f'{p:.3f}'
FR={'0.05':'05','0.1':'10','0.25':'25','1.0':'100'}
g=o['graph_vs_empty']
for fr,t in FR.items():
    for met,m in (('auc','a'),('new','n')):
        for a,b,nm in (('P/support','P/empty','sup'),('P/cag','P/support','cagsup'),('P/cag','P/empty','cag'),('P/gate2','P/support','g2sup'),('P/gate2','P/empty','g2'),
                       ('P/support-pooled','P/empty','pool'),('P/support-pooled','P/support','poolsup')):
            k=f'{fr}|{met}|{a}|{b}'
            if k not in g: continue
            r=g[k]; base=f'{nm}{t}{m}'
            S[base]=sg(r['mean']); S[base+'_pos']=f"{r['pos']}"; S[base+'_n']=f"{r['n']}"; S[base+'_p']=pv(r['p']); S[base+'_ph']=pv(r['p_holm'])
            S[base+'_med']=sg(r['median']); S[base+'_min']=sg(r['min']); S[base+'_max']=sg(r['max'])
dr=o['dose_response']
for met,m in (('auc','a'),('new','n')):
    q=dr[f'{met}|P/support|P/empty']['quadratic']; w=dr[f'{met}|P/support|P/empty']['within_fraction_spearman']
    S[f'quad{m}_p']=pv(q['lr_p']); S[f'quad{m}_peak']=f"{q['peak_ipk']:.0f}"; S[f'quad{m}_b2']=sg(q['b2'])
    for fr,t in FR.items():
        if fr in w: S[f'rho{t}{m}']=sg(w[fr]['rho'],2); S[f'rho{t}{m}_p']=pv(w[fr]['p'])
    S[f'lin{m}_slope']=sg(dr[f'{met}|P/support|P/empty']['slope']); S[f'lin{m}_p']=pv(dr[f'{met}|P/support|P/empty']['p'])
gq=o['graph_quality']
for fr,t in (('0.05','05'),('0.1','10'),('0.25','25')):
    S[f'rec{t}']=f"{100*np.mean([v['recall_of_full_edges'] for k,v in gq.items() if k.endswith('|'+fr)]):.0f}"
    S[f'edg{t}']=f"{100*np.mean([v['edges']/v['full_edges'] for k,v in gq.items() if k.endswith('|'+fr)]):.0f}"
ab=o['ablation']
for k,nm in (('gru','gru'),('att','att'),('same','same'),('hz','hz'),('r1_none_vs_empty','r1none')):
    r=ab[k]; S[f'abl_{nm}']=sg(r['mean']); S[f'abl_{nm}_abs']=f"{abs(r['mean']):.3f}"; S[f'abl_{nm}_med']=sg(r['median']); S[f'abl_{nm}_neg']=f"{r['n']-r['pos']}"; S[f'abl_{nm}_n']=f"{r['n']}"; S[f'abl_{nm}_ph']=pv(r['p_holm']); S[f'abl_{nm}_min']=sg(r['min'])
pc=o['pc_loss']
for k,nm in (('graded|auc','pcga'),('graded|new','pcgn'),('permuted|auc','pcpa'),('permuted|new','pcpn'),('graded-permuted|auc','pcda'),('graded-permuted|new','pcdn')):
    r=pc[k]; S[nm]=sg(r['mean']); S[nm+'_pos']=f"{r['pos']}"; S[nm+'_ph']=pv(r['p_holm']); S[nm+'_p']=pv(r['p'])
v=o['r1_inputs_vs_empty']
for k,r in v.items():
    c,met=k.split('|'); nm='r1_'+c.split('/')[1].replace(':','_')+('_a' if met=='auc' else '_n')
    S[nm]=sg(r['mean']); S[nm+'_pos']=f"{r['pos']}"; S[nm+'_p']=pv(r['p']); S[nm+'_ph']=pv(r['p_holm'])
cm=o['model_comparisons']
for k,r in cm.items():
    fr,a,b=k.split('|'); nm=f"cmp{FR[fr]}_{a.split('/')[1]}_{b}"
    S[nm]=sg(r['mean'],3); S[nm+'_pos']=f"{r['pos']}"; S[nm+'_n']=f"{r['n']}"; S[nm+'_ph']=pv(r['p_holm']); S[nm+'_min']=sg(r['min'],3); S[nm+'_max']=sg(r['max'],3)
S['sub_sd10']=f"{o['sub_sd']['0.1']:.4f}"; S['sub_sd05']=f"{o['sub_sd']['0.05']:.4f}"
S['n_runs_r2']=str(o['n_runs']['R2']); S['n_runs_r1']=str(o['n_runs']['R1'])
# gate values (stored for runs trained after the gate logging was added)
import glob
G={'g1':[],'g2':[]}
for f in glob.glob(f'{K}/runs/*_g*__s*.json'):
    j=json.load(open(f))
    if isinstance(j.get('gate'),dict): G['g2' if '_g2' in j['tag'] else 'g1'].append(j['gate'])
for k,L in G.items():
    if L:
        S[f'{k}_mean']=f"{np.mean([x['g_mean'] for x in L]):.2f}"; S[f'{k}_lo']=f"{np.mean([x['g_min'] for x in L]):.2f}"; S[f'{k}_hi']=f"{np.mean([x['g_max'] for x in L]):.2f}"
        S[f'{k}_w1']=sg(float(np.mean([x['w1'] for x in L])),3); S[f'{k}_nruns']=str(len(L))
# Junyi expert map against the log-built support graph (10%, sub 0)
m=runs[(runs.frac==0.1)&(runs['sub']==0)&(runs.data=='junyi')].groupby('cfg').auc.mean()
S['jy_expert']=f"{m['R1/transition:expert']:.4f}"; S['jy_support']=f"{m['R1/transition:support']:.4f}"; S['jy_empty']=f"{m['P/empty']:.4f}"
def cell(cfg,d,fr):
    x=runs[(runs.cfg==cfg)&(runs.data==d)&(runs.frac==fr)&(runs['sub']==0)].auc
    return f'{x.mean():.4f}' if len(x) else '--'
def main_table(fr):
    rows=[('DKT','DKT'),('SAKT','SAKT'),('AKT','AKT'),('GKT','GKT'),('simpleKT','simpleKT'),('RKT','RKT'),('FKG-KT, empty graph','P/empty'),('FKG-KT, support graph','P/support'),('FKG-KT + CAG','P/cag')]
    h='| Model | '+' | '.join(NAMES)+' |\n|'+'---|'*(len(NAMES)+1)+'\n'
    return h+'\n'.join('| '+n+' | '+' | '.join(cell(c,d,fr) for d in D10)+' |' for n,c in rows)
S['TABLE2_SPARSE']=main_table(0.1); S['TABLE2_FULL']=main_table(1.0)
def lc_table():
    h='| Training learners | Interactions per KC (median over corpora) | Support − empty, all | corpora > 0 | Holm p | Support − empty, first attempts | corpora > 0 | Holm p |\n|---|---|---|---|---|---|---|---|\n'
    out=[]
    for fr,t,lab in (('0.05','05','5% (5 subsamples)'),('0.1','10','10% (5)'),('0.25','25','25% (3)'),('1.0','100','100% (1)')):
        ip=runs[(runs.frac==float(fr))&runs.data.isin(D10)].groupby(['data','sub']).ipk.first().groupby('data').mean().median()
        out.append(f"| {lab} | {ip:.0f} | {S[f'sup{t}a']} | {S[f'sup{t}a_pos']}/10 | {S[f'sup{t}a_ph']} | {S[f'sup{t}n']} | {S[f'sup{t}n_pos']}/10 | {S[f'sup{t}n_ph']} |")
    return h+'\n'.join(out)
S['TABLE3_LC']=lc_table()
def inputs_table():
    rows=[('order-preserving: support','transition_support'),('order-preserving: permuted grades','transition_permuted'),('order-preserving: graded','transition_graded'),('order-preserving: rank-coded','transition_rank'),
          ('order-preserving: lower-quartile cut','transition_cut25'),('order-preserving: median cut','transition_cut50'),('order-preserving: upper-quartile cut','transition_cut75'),('order-preserving: product closure, 3 hops','transition_product3'),
          ('co-practice: support','copractice_support'),('co-practice: rank-coded','copractice_rank'),('co-practice: permuted grades','copractice_permuted'),('neural: DKT influence','dktinf_rank')]
    h='| Graph input (10% of learners, sample 0) | AUC diff. | corpora > 0 | Holm p | first-attempt diff. | corpora > 0 | Holm p |\n|---|---|---|---|---|---|---|\n'
    return h+'\n'.join(f"| {n} | {S['r1_'+k+'_a']} | {S['r1_'+k+'_a_pos']}/8 | {S['r1_'+k+'_a_ph']} | {S['r1_'+k+'_n']} | {S['r1_'+k+'_n_pos']}/8 | {S['r1_'+k+'_n_ph']} |" for n,k in rows)
S['TABLE4_INPUTS']=inputs_table()
def corpus_table():
    t1=json.load(open(f'{K}/paper/r2_table1.json'))
    h='| Corpus | Learners | Interactions | KCs | Correct rate | Train. int. per KC | Edges (order-pres., all learners) | Expert structure |\n|---|---|---|---|---|---|---|---|\n'
    ex={'junyi':'prerequisite map','algebra05':'units','eedi':'subject tree','ednet':'TOEIC parts','bridge06':'units','statics11':'units'}
    lab=dict(zip(D10,['ASSISTments 2009','ASSISTments 2012','ASSISTments 2015','ASSISTments 2017','Junyi Academy','Algebra 2005','Eedi','EdNet KT1','Bridge to Algebra 2006','Statics 2011']))
    return h+'\n'.join(f"| {lab[r[0]]} | {r[1]:,} | {r[2]:,} | {r[3]} | {r[5]:.3f} | {r[6]:.0f} | {r[8]:,} | {ex.get(r[0],'--')} |" for r in t1)
S['TABLE1']=corpus_table()
def secs_table():
    sec=runs[(runs.src=='R2')&runs.data.isin(['bridge06','statics11'])&(runs['sub']==0)].groupby(['cfg','frac']).secs.mean()   # all CPU, same two corpora
    rows=[('DKT','DKT'),('SAKT','SAKT'),('AKT','AKT'),('GKT','GKT'),('simpleKT','simpleKT'),('RKT','RKT'),('FKG-KT, empty graph','P/empty'),('FKG-KT, support graph','P/support'),('FKG-KT + CAG','P/cag')]
    h='| Model | 10% of learners | 100% of learners |\n|---|---|---|\n'
    def v(c,f):
        x=sec.get((c,f),float('nan')); return '--' if x!=x else f'{x:.0f}'
    return h+'\n'.join(f"| {n} | {v(c,0.1)} | {v(c,1.0)} |" for n,c in rows)
S['TABLE5_SECS']=secs_table()
class Strict(dict):
    def __missing__(s,k): raise KeyError(f'manuscript asks for {k!r}, which r2_stats.py does not compute')
def render(text): return string.Formatter().vformat(text,(),Strict(S))
# ---- second round: pyKT protocol and matched-density DKT influence (r2_extra.json)
import os
if os.path.exists(f'{K}/paper/r2_extra.json'):
    X=json.load(open(f'{K}/paper/r2_extra.json'))
    for k,v in X.get('pykt',{}).items():
        if k=='pykt_pub': continue
        d,mn=k.split('|'); nm=f"pk_{d}_{mn.replace('-','_')}"
        if not isinstance(v,dict): S[nm]=f"{v:.0f}"; continue
        if 'sd' in v:
            S[nm]=f"{v['mean']:.4f}"; S[nm+'_sd']='--' if v['sd'] is None else f"{v['sd']:.4f}"; S[nm+'_n']=str(v['n'])
            if v.get('published'): S[nm+'_pub']=f"{v['published']:.4f}"; S[nm+'_pubsd']=f"{v['published_sd']:.4f}"
        elif 'pos' in v: S[nm]=sg(v['mean']); S[nm+'_pos']=str(v['pos']); S[nm+'_n']=str(v['n'])
        else: S[nm]=f"{v:.0f}"
    q=X['dkteq']
    for k,v in q.items():
        if not isinstance(v,dict) or 'mean' not in v: continue
        nm='eq_'+k.replace(':','_').replace('|','_').replace('-','_')
        S[nm]=sg(v['mean']); S[nm+'_pos']=str(v['pos']); S[nm+'_n']=str(v['n']); S[nm+'_ph']=pv(v['p_holm'])
    e=q['edges']; S['eq_edges_med']=f"{np.median([v['op'] for v in e.values()]):.0f}"; S['eq_dkt10_med']=f"{np.median([v['dkt10'] for v in e.values()]):.0f}"
    S['eq_ratio_med']=f"{np.median([v['dkt10']/v['op'] for v in e.values()]):.0f}"
    for k,(a,b) in X.get('pykt_pub',{}).items():
        d,mn=k.split('|'); S[f'pk_{d}_{mn}_pub']=f"{a:.4f}"; S[f'pk_{d}_{mn}_pubsd']=f"{b:.4f}"
    _pc=json.load(open(f'{K}/paper/r2_results.json'))['graph_vs_empty']['1.0|auc|P/support|P/empty']['per_corpus']
    S['full_sup_statics11']=sg(_pc['statics11']); S['full_sup_assist15']=sg(_pc['assist15'])
    def pk_table():
        PB=X.get('pykt_pub',{})
        rows=[('DKT (pyKT code)','dkt'),('SAKT (pyKT code)','sakt'),('AKT (not rerun)','akt'),('simpleKT (pyKT code)','simplekt'),('DKT (our implementation)','ourdkt'),('FKG-KT, empty graph','fkg_empty'),('FKG-KT, support graph','fkg_support')]
        dss=[d for d in ('statics2011','assist2015') if any(k.startswith(d+'|') for k in X.get('pykt',{}))]
        lab={'statics2011':'Statics 2011','assist2015':'ASSISTments 2015'}
        h='| Model | '+' | '.join(f'{lab[d]}: this run | {lab[d]}: published' for d in dss)+' |\n|'+'---|'*(1+2*len(dss))+'\n'
        out=[]
        for n,mn in rows:
            cells=[]
            for d in dss:
                v=X['pykt'].get(f'{d}|{mn}')
                cells.append('--' if v is None else (f"{v['mean']:.4f} ± {v['sd']:.4f} ({v['n']})" if v['sd'] is not None else f"{v['mean']:.4f} (1)"))
                pb=PB.get(f'{d}|{mn}')
                cells.append(f"{pb[0]:.4f} ± {pb[1]:.4f}" if pb else '--')
            out.append('| '+n+' | '+' | '.join(cells)+' |')
        return h+'\n'.join(out)
    S['TABLE6_PYKT']=pk_table()

if __name__=='__main__':
    for k in sorted(S):
        if not k.startswith('TABLE'): print(k,S[k])
