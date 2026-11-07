"""R2 job list, in priority order. Usage: genjobs.py phase corpus..."""
import sys
ph=sys.argv[1]; C=sys.argv[2:]; J=[]
def fk(c,f,s,seed,extra): J.append(f"--data {c} --frac {f} --sub {s} --seed {seed} --model fkgkt3 {extra}")
CFG=['--graph empty','--graph support','--graph support --gate 1']
NEW={'statics11','bridge06'}
if ph=='lc_low':           # learning curve, low-data end
    for f,ns in ((0.05,5),(0.1,5),(0.25,3)):
        for c in C:
            for s in range(ns):
                for g in CFG: fk(c,f,s,s,g)
if ph=='abl':
    for c in C:
        for s in range(3):
            for a in ('gru','att','same','hz'): fk(c,0.1,s,s,f'--graph empty --abl {a}')
if ph=='pc':
    for c in C:
        for s in range(3):
            for w in ('graded','permuted'):
                for l in (0.1,): fk(c,0.1,s,s,f'--graph empty --pc {w} --pclam {l}')
if ph=='base_sparse':
    for c in C:
        ms=['simplekt','rkt']+(['dkt','sakt','akt','gkt'] if c in NEW else [])
        for m in ms:
            for s in range(3): J.append(f"--data {c} --frac 0.1 --seed {s} --model {m}")
if ph=='lc_high':
    for c in C:
        for s in range(1):
            for g in CFG: J.append(f"--data {c} --frac 1.0 --seed {s} --model fkgkt3 {g}")
if ph=='base_full':
    for c in C:
        ms=[('simplekt',1),('rkt',1)]+([('dkt',1),('sakt',1),('akt',1)]+([('gkt',1)] if c!='bridge06' else []) if c in NEW else [])
        for m,n in ms:
            for s in range(n): J.append(f"--data {c} --frac 1.0 --seed {s} --model {m}")
print('\n'.join(J))
