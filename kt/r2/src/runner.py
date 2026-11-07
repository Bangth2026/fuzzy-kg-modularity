import sys, subprocess, os, itertools, json
from concurrent.futures import ThreadPoolExecutor
jobs=[l.split() for l in open(sys.argv[1]) if l.strip()]
W=int(sys.argv[2]) if len(sys.argv)>2 else 2
def run(j):
    r=subprocess.run(['python3','/home/claude/kt/src/train.py']+j,capture_output=True,text=True,env={**os.environ,'NT':'1'})
    with open('/home/claude/kt/runs/runner.log','a') as f: f.write((r.stdout.strip() or ('ERR '+' '.join(j)+' '+r.stderr[-500:]))+'\n')
with ThreadPoolExecutor(W) as ex: list(ex.map(run,jobs))
