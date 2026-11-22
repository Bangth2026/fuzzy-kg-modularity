"""Release 3.12.0: each claim of Table VI against delta in {0.001, 0.002, 0.003}.

Classes, from the 90% interval (two one-sided tests at alpha = 0.05):
  B  beyond delta: the interval lies wholly above +delta (or below -delta);
  S  small: the interval excludes 0 and lies inside (-delta, +delta), so the term is detected and equivalent to zero;
  N  nonzero: the interval excludes 0 and reaches past +/-delta without lying beyond it;
  E  equivalent to zero: the interval includes 0 and lies inside (-delta, +delta);
  U  unresolved: the interval includes 0 and reaches past +/-delta.
Intervals recomputed here (source r15) come from results_r15/r15_results.json; the others are the 90% intervals of
release 3.11.0 as reported in Supplementary Section S9 (source S9), whose run files are in kt/runs_r14.
Writes results_r15/r15_delta.json and prints the grid.
"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); OUT = f'{REPO}/results_r15'
R = json.load(open(f'{OUT}/r15_results.json'))
DELTAS = (0.001, 0.002, 0.003)


def r15(block, key, k='first'):
    o = R[block][key][k]; return o['mixed'], o['ci90'][0], o['ci90'][1], 'r15'


CLAIMS = [
    ('Graph gain, both entry points, 10%, design (32 units)', r15('terms', 'both entry points G - empty (runs_r6)')),
    ('Graph gain, both entry points, held out (H1)', (0.0025, 0.0018, 0.0030, 'S9.20')),
    ('Graph gain, both entry points, 12 corpora pooled', (0.0040, 0.0025, 0.0045, 'S9.20')),
    ('Graph gain, all training learners, three seeds', (0.0007, -0.0004, 0.0018, 'S9.17')),
    ('Embedding: graph gain G - empty', r15('terms', 'emb G - empty')),
    ('Embedding: degree term DP - ER', r15('terms', 'emb DP - ER')),
    ('Embedding: edge term G - DP', r15('terms', 'emb G - DP')),
    ('Embedding: G - RV, design', (0.0008, -0.0020, 0.0031, 'S9.3')),
    ('Embedding: G - W2, design', (0.0008, -0.0016, 0.0026, 'S9.5')),
    ('Embedding: G - RV, held out (H2b)', (0.0016, 0.0011, 0.0020, 'S9.20')),
    ('Embedding: G - RV, 12 corpora pooled', (0.0009, -0.0013, 0.0028, 'S9.20')),
    ('State: random graph ER - empty, design', r15('terms', 'state ER - empty')),
    ('State: random graph ER - empty, held out (H4)', (0.0000, -0.0003, 0.0003, 'S9.20')),
    ('State: edge term G - DP, design', r15('terms', 'state G - DP')),
    ('State: edge term G - DP, held out (H3)', (0.0003, 0.0002, 0.0005, 'S9.20')),
    ('State: edge term G - DP, 12 corpora pooled', (0.0013, 0.0007, 0.0020, 'S9.20')),
    ('State: G - UR (six corpora)', (0.0029, 0.0009, 0.0061, 'S9.13')),
    ('State: UR - ER (six corpora)', (0.0000, -0.0020, 0.0014, 'S9.13')),
    ('Entry point x edge term, first attempts', r15('interaction', 'edges: (G - DP)_state - (G - DP)_emb')),
    ('Entry point x edge term, all predictions', r15('interaction', 'edges: (G - DP)_state - (G - DP)_emb', 'all')),
    ('Edge term, embedding, all predictions', r15('terms', 'emb G - DP', 'all')),
    ('Edge term, state, all predictions', r15('terms', 'state G - DP', 'all')),
    ('GKT: edge term G - DP', (0.0248, 0.0121, 0.0385, 'S9.15')),
    ('GKT: G - U', (-0.0218, -0.0393, -0.0035, 'S9.15')),
]


def cls(lo, hi, d):
    if lo > d or hi < -d: return 'B'
    nz = lo > 0 or hi < 0; inside = -d < lo and hi < d
    return ('S' if inside else 'N') if nz else ('E' if inside else 'U')


out = []
for lab, (est, lo, hi, src) in CLAIMS:
    out.append(dict(claim=lab, estimate=est, ci90=[lo, hi], source=src, **{str(d): cls(lo, hi, d) for d in DELTAS}))
    print(f"{lab:52s} {est:+.4f} [{lo:+.4f}, {hi:+.4f}] {src:6s} " + ' '.join(cls(lo, hi, d) for d in DELTAS))
json.dump(out, open(f'{OUT}/r15_delta.json', 'w'), indent=1)
