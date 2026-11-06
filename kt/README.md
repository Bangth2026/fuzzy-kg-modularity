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

# Release 2.1.0: controls of revision R5

Adds the runs of Section 6.6 of the revised manuscript (recency and degree-preserving controls), the
rerun empty-graph control of Table 4, a repetition of the support-versus-control comparison at 5%,
10%, 25% and 100% of the training learners, one cell with a larger learning rate for the graph
scalars, and two corpora (Bridge to Algebra 2006, Statics 2011). The model is `models.FKGKT2` of
release 2.0.0; `r5_controls.FKGKT3` only adds the optional recency term and logs the learned scalars.
Seed 0 on sample 0 of the 10% regime reproduces the archived run (Algebra 2005, support graph:
test AUC 0.7379, first attempts 0.7088).

| Step | Script | Output |
|---|---|---|
| Preprocess (adds `bridge06`, `statics`; raw files from base.ustc.edu.cn/data/) | `src/prep.py a09 a15 alg05 statics bridge06 eedi ednet a12 a17` | `data/<corpus>.parquet` |
| Junyi with little memory (same learners as release 2.0.0) | `src/prep_junyi_lowmem.py` | `data/junyi.parquet` |
| One run (graph of the subsample built on the fly) | `src/r5_controls.py --data <c> --frac <f> --sub <s> --cond {empty,support,recency,suprec,rewired} [--seed k] [--lr_scalar x]` | `runs_r5/<tag>.json`, `_pred.npz`; `graphs_r5/` |
| Queue | `src/r5_runner.py jobs_a.txt 2` (then `jobs_b.txt`, `jobs_c.txt`) | |
| Table 5, learning-curve repetition, mixed model with leave-one-corpus-out, diagnostics | `src/r5_analyse.py` | `results_r5/r5_results.json` |
| Table 4 (graph inputs of release 2.0.0 against the rerun control), per-seed Algebra 2005 and Eedi | `src/r5_table4.py` (reads `results/runs/`) | `results_r5/table4_r5.json` |

Every run file records test AUC on all predictions and on first attempts, ACC, RMSE, ECE (15 bins),
the number of first-attempt predictions, the share of first attempts whose preceding KC is a
prerequisite of the target in the support graph, epochs run and whether the 30-epoch cap was hit,
the learned beta_pre, beta_suc, beta_same, beta_rec and the mean norm of the graph message.
Subsample s >= 1 is drawn with seed 12345 + s, without replacement and independently of the others;
subsample 0 is the 10% sample of release 2.0.0. Prediction files (`*_pred.npz`) are not included
(size); their SHA-256 checksums are in `results_r5/pred_checksums_sha256.txt`.
