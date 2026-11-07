# Release 2.0.0: fuzzy prerequisite KGs for neural knowledge tracing

Adds the knowledge-tracing experiments of the Neurocomputing manuscript.

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

# Release 2.1.0: recency and degree-preserving controls

Adds the runs of Section 6.6 of the manuscript (recency and degree-preserving controls), the
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

# Release 3.1.0: training budget by updates, and two further graph controls

Adds `src/r6_controls.py` (training and evaluation), `src/r6_analyse.py` (all statistics), run files in
`runs_r6/` and results in `results_r6/r6_results.json`. The model and data pipeline are those of
`src/r5_controls.py`; with `--budget epochs` and an existing condition the script reproduces the run
of release 2.1.0 exactly (Statics 2011, 10%, subsample 0, support graph: test AUC 0.7450).

| Step | Script | Output |
|---|---|---|
| One run, budget by epochs (30 epochs, patience 3) or by updates (at most 2,000 updates, validation every 25, patience 8 evaluations) | `src/r6_controls.py --data <c> --frac <f> --sub <s> --cond {empty,support,rewired,er,degfeat} --budget {epochs,steps}` | `runs_r6/<tag>.json`, `_pred.npz` |
| Queue (two workers) | `src/r6_runq.sh runs_r6/jobs/jobs_e.txt` (job lists of the original run in `runs_r6/jobs/`, run in the order a, c, d, e; finished runs are skipped) | |
| Update counts of every learning-curve run, pooled support-versus-control comparison (kt/r2 runs and runs_r5), bootstrap and mixed-model intervals, step-budget table, controls table, run counts by set | `src/r6_analyse.py` | `results_r6/r6_results.json` |

Conditions added: `er`, a directed Erdős–Rényi graph with as many edges as the support graph of the
subsample (uniform over ordered pairs, no self-loops, no reciprocal pairs, seed 2000 + s); `degfeat`,
the empty graph plus the standardised log in- and out-degree of every KC in the support graph,
mapped by a learned linear layer into the KC embedding before the layer normalisation.
Subsamples are drawn as in release 2.1.0 (subsample s >= 1 with seed 12345 + s). The learning-curve
runs of `kt/r2` draw subsample s >= 1 with seed 12345 + 1000 s, so the two sets share subsample 0
only; the pooled comparison averages the two runs on subsample 0. Prediction files are not
included (size); their SHA-256 checksums are in `results_r6/pred_checksums_sha256.txt`.

# Release 3.2.0: zero-initialised degree feature, complete random-graph controls

Adds the condition `degfeat0` to `src/r6_controls.py`: the degree-feature control with its linear map
initialised at zero (weight and bias), so that at the first update the model is the empty-graph
control and departs from it only through training. With `degfeat` (release 3.1.0) the map has
PyTorch's default random initialisation and adds a fixed random vector to every KC embedding, which
the tracers on the smallest corpora, with a few dozen updates, do not undo. Also adds the Erdős–Rényi
control on subsamples 3 and 4 at 10% under the epoch budget (so that it is paired with every
subsample of release 2.1.0), and, under the update budget, the Erdős–Rényi control on the six larger
corpora and the degree-preserving control on their subsample 1. Every other condition and run file is
unchanged; with an existing condition the script reproduces the archived runs (Statics 2011, 10%,
subsample 0, support graph: 0.7450 under the epoch budget, 0.7557 under the update budget; EdNet,
update budget, 0.6736; ASSISTments 2012, epoch budget, 0.7049).

| Step | Script | Output |
|---|---|---|
| Preprocessing: `src/prep.py` with one corpus per call (`a12` and `ednet` draw their learners from a fresh generator); Junyi with `src/prep_junyi_lowmem.py` | | `data/<corpus>.parquet` |
| Runs of this release (job list in `runs_r6/jobs/jobs_f.txt`) | `src/r6_runq.sh runs_r6/jobs/jobs_f.txt` | `runs_r6/<tag>.json` |
| Controls table at 10% (lower block), update-budget table with rows of four corpora reported as descriptive and outside the Holm family, comparisons of the graph controls pooled over both budgets with bootstrap and mixed-model intervals, recency control with intervals, learned scalars under the update budget, share of first attempts by corpus, run counts by set | `src/r32_analyse.py` | `results_r6/r32_results.json`, `results_r6/run_counts.csv` |

`results_r6/r6_results.json` (release 3.1.0) is left as it was.
