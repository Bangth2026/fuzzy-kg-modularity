"""Version 4.6.0. Steps 2d to 3 of Algorithm 2 on PPI5k (global and row weights-only nulls, 50 draws,
three seeds; power with 10 draws per tilt; cut audit over ten seeds), run on a second core beside
scripts/v9_audit_cost.py, which computes the floor.  Writes results/v9/audit_ppi5k_rest.json."""
import json, time, sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from fuzzykg import FuzzyGraph
from fuzzykg.audit import cut_audit
from fuzzykg.nulltest import best_qf, weights_only_test, weights_only_power
e=pd.read_csv('data/graphs/ppi5k_edges.csv'); G=FuzzyGraph.from_edges(zip(e.src,e.dst,e.mu))
seeds=(0,1,2); obs=best_qf(G,1.0,seeds); out={}
def save(): json.dump(out,open('results/v9/audit_ppi5k_rest.json','w'),indent=1)
t=time.perf_counter(); wg=weights_only_test(G,'global',50,1.0,seeds,obs=obs.Q_F); out['step2d_global_null']={'seconds':round(time.perf_counter()-t,2),'z':wg['z']}; save(); print(out,flush=True)
t=time.perf_counter(); wr=weights_only_test(G,'row',50,1.0,seeds,obs=obs.Q_F); out['step2e_row_null']={'seconds':round(time.perf_counter()-t,2),'z':wr['z']}; save(); print(out,flush=True)
t=time.perf_counter(); pg=weights_only_power(G,obs.membership,kind='global',draws=10); out['step2f_power_global']={'seconds':round(time.perf_counter()-t,2),'r':pg['detectable_r_pb']}; save()
t=time.perf_counter(); pr=weights_only_power(G,obs.membership,kind='row',draws=10); out['step2g_power_row']={'seconds':round(time.perf_counter()-t,2),'r':pr['detectable_r_pb']}; save()
cuts={f"q{q:.2f}": float(np.quantile(G.mu,q)) for q in (0.25,0.5,0.75)}
t=time.perf_counter(); ca=cut_audit(G,cuts,seeds=range(10)); out['step3_cut_audit']={'seconds':round(time.perf_counter()-t,2),'cuts':{c:{k:r.get(k) for k in ('qf_drop','qf_drop_sd','rank_drop','rank_drop_sd')} for c,r in ca['cuts'].items()}}; save(); print('done',flush=True)
