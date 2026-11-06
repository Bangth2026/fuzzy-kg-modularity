# Release 2.0.0: fuzzy prerequisite KGs for neural knowledge tracing

Adds the experiments of the Neurocomputing revision.

| Step | Script | Output |
|---|---|---|
| Download public corpora (USTC EduData mirror) and preprocess: 8 corpora, <=6,000 learners, first 200 interactions | `src/prep.py a09 a12 a15 a17 junyi alg05 eedi ednet` | `data/<corpus>.parquet` |
| Fuzzy KGs from training learners (order-preserving, co-practice), candidate graphs, audits; `FRAC=0.1` for the sparse regime | `src/graphs.py <corpora>` | `graphs/<corpus>[_f10]_{transition,copractice}.npz`, `_audit.json` |
| Expert structures | `src/expert.py junyi eedi algebra05 ednet` | `graphs/<corpus>_expert.npy` |
| DKT, SAKT, GKT, FKG-KT training (CPU) | `src/train.py --data <c> --frac <f> --model {dkt,sakt,gkt,fkgkt2} [--construction ..] [--graph ..] --seed s`; queue runner `src/runner.py` | `runs/<tag>.json`, `_pred.npz` |
| DKT-influence (neural) KG | `src/dktgraph.py <corpora>` (after DKT seed 0) | `graphs/<corpus>_dktinf.npz` |
| Algorithm 1 on the new graphs | `src/nulltests.py <corpora>` | `results/graphs/nulltests_all.json` |
| Lemma 1 covariance term, Proposition 1 bound and certificate, counterexamples | `theory/covbound.py`, `theory/covbound_new.py`, `theory/counterexample.py` | `theory/*.json` |
| Tables, bootstrap, Wilcoxon, figures | `src/final.py`, `src/figures.py`, `assemble.py` | `results/results.json` |

Per-run JSON files of all 527 training runs are in `results/runs/`. Prediction files are not included (size); `train.py` regenerates them.
