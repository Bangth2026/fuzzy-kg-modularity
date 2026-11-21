"""Version 4.6.0. Merge results/v9/audit_ppi5k_rest.json (scripts/v9_ppi_parallel_steps.py) into
results/v9/audit_cost_ppi5k.json (scripts/v9_audit_cost.py ppi5k: steps 1 to 2c)."""
import json
R="results/v9/"
a=json.load(open(R+"audit_cost_ppi5k.json")); b=json.load(open(R+"audit_ppi5k_rest.json"))
keep={k:v for k,v in a["steps"].items() if k in ("load","step1_describe","one_leiden_run","step2a_observed_partition","step2b_degree_null","step2c_floor")}
keep.update(b); a["steps"]=keep
a["note"]=("steps 2d to 3 computed by scripts/v9_ppi_parallel_steps.py on a second core; step 2b from the first run of "
           "scripts/v9_audit_cost.py, step 2c from the run with --resume; timings of PPI5k are not reported because the core was shared")
for k in ("total_seconds","algorithm1_seconds","audit_without_algorithm1_seconds"): a.pop(k,None)
json.dump(a,open(R+"audit_cost_ppi5k.json","w"),indent=1); print(list(a["steps"]))
