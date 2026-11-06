import sys, subprocess, os
from concurrent.futures import ThreadPoolExecutor
jobs=[l.split() for l in open(sys.argv[1]) if l.strip()]
W=int(sys.argv[2]) if len(sys.argv)>2 else 2
here=os.path.dirname(os.path.abspath(__file__))
def run(j):
    r=subprocess.run(['python3',f'{here}/r5_controls.py']+j,capture_output=True,text=True,env={**os.environ,'NT':'1'})
    with open(os.environ.get('KT_ROOT','/home/claude/kt')+'/runs_r5/runner.log','a') as f: f.write((r.stdout.strip() or ('ERR '+' '.join(j)+' '+r.stderr[-800:]))+'\n')
with ThreadPoolExecutor(W) as ex: list(ex.map(run,jobs))
