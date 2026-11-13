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

# Release 3.3.0: channel-separating controls and analyses of the training budget

Adds four conditions to `src/r6_controls.py` that read a graph at one of the probe's two entry points
only: `support_emb` and `er_emb` (the graph in the embedding refinement of Eq. (2), with the
prerequisite and successor terms of the attention bias of Eq. (3) given a zero matrix) and
`support_att` and `er_att` (the graph in the attention bias only, with the neighbour matrices of
Eq. (2) set to zero). Each variant is built from the full-graph model and then has one matrix
zeroed, so its initial weights are those of the full-graph run with the same graph. 128 runs under
the update budget at 10% (job list `runs_r6/jobs/jobs_g.txt`).

`src/r33_analyse.py` writes `results_r6/r33_results.json`, `results_r6/table6_upper_by_corpus.csv`
and `results_r6/checkpoint_updates.csv`: the rise of the graph gain from 5% to 10% under the two
budgets on the same corpus x subsample units, with a bootstrap interval for the difference of the
rises; the updates to the selected checkpoint by corpus under the two budgets; the per-corpus values
of the upper rows of Table 6; the graph controls pooled over both budgets with a fixed effect for
the budget; and the channel-separating controls. These are the tables of Supplementary Section S8.
Earlier results files are unchanged.

# Release 3.4.0: robustness of the main comparisons and decision-level metrics

Adds `src/r34_analyse.py`, which reads the archived run files only (no new training runs) and writes
`results_r6/r34_results.json`:

- for the main comparisons (the support graph against the empty-graph control, pooled over the two
  sets of epoch-budget runs at 10% and 25%, and under the update budget at 10%; the recency,
  degree-preserving and Erdos-Renyi controls on first attempts at 10%): the mean of corpus means, a
  corpus-then-subsample bootstrap interval, a mixed-model interval (random intercept per corpus) and
  leave-one-corpus-out summaries (range of the mean, smallest number of corpora with a positive
  difference, largest Wilcoxon p over the nine remaining corpora);
- accuracy at the 0.5 threshold, RMSE and ECE of the support graph against the control on all test
  predictions, by training budget and size;
- the number of test predictions and first attempts per corpus at 10%;
- the expected overlap between two subsamples of one corpus (m^2 / N learners, a share m / N of each).

- refits of the learning-curve mixed model (random slope; regressor centred within corpus; without
  the 100% cell);
- the learned attention scalars in the channel-separating runs of release 3.3.0.

These are the values of Table 9 and Sections 6.2, 6.7 and 6.10 of the manuscript. The mixed model needs
`statsmodels` (added to `requirements.txt`). Run with `KT_ROOT=<checkout>/kt python kt/src/r34_analyse.py`.

New runs of this release (update budget, 10% of the training learners, the subsamples of Table 6:
five on Statics 2011, Algebra 2005, Bridge to Algebra and ASSISTments 2017, two on the other six):

| Step | Script | Output |
|---|---|---|
| One run | `src/r7_controls.py --data <c> --frac 0.1 --sub <s> --cond {empty,support,uniform_emb,empty_l2,support_l2} [--l2 x] [--lr x]` | `runs_r7/<tag>.json`, `_pred.npz` |
| Queue (two workers; job list `runs_r7/jobs/jobs_h.txt`) | `src/r7_runq.sh runs_r7/jobs/jobs_h.txt` | |
| Analysis | `src/r7_analyse.py` | `results_r7/r7_results.json` |

Conditions: `uniform_emb`, the complete graph read in the embedding refinement only (every KC mixed with
the mean embedding of all other KCs; the attention bias gets a zero matrix); `empty_l2` and
`support_l2`, the control and the support graph with the penalty lambda * mean_k ||E_k - mean_j E_j||^2
on the KC embedding table, lambda in {1e-4, 1e-3, 1e-2, 1e-1}; `--lr 0.005` and `--lr 0.01` for the
control and the support graph (default 0.002). On the six larger corpora the shrinkage grid is
{1e-3, 1e-2, 1e-1} and both grids were run on subsample 0 only; 396 runs in all. Per-prediction files keep the target KC, which the
analysis uses to split first attempts by the training interactions on the target KC (terciles within
corpus), to decompose first-attempt AUC into within-KC and between-KC parts, and to score a trivial
predictor (training-set correct rate of the target KC). `empty` and `support` reproduce the archived
runs of release 3.1.0 on the same subsamples. Prediction files are not included (size); their SHA-256
checksums are in `results_r7/pred_checksums_sha256.txt`. Raw corpora: the Junyi archive on the EduData
mirror holds a problem log over 2 GB that some RAR extractors cannot unpack; `unrar` works.
Earlier results files are unchanged.

# Release 3.7.0: neighbour split for the frozen table, position in the sequence, run inventory

Analyses of the per-prediction files in `runs_r9/` (release 3.6.0); no model is trained and no data file is read.
`src/r10_analyse.py` writes `results_r10/r10_results.json`:

1. First attempts split by whether an in- or out-neighbour of the target KC (Erdos-Renyi graph `er`, support graph `sup`)
   occurs earlier in the learner's sequence, for `support_emb`, `er_emb` and `erR_emb` against `empty`, and for the contrast
   `er_emb - erR_emb` (learned against frozen neighbour table over the same graph), on each group and seen - unseen within
   unit (a group enters a unit only with more than 20 first attempts and both outcome classes, as in `r9_analyse.py`).
   The `er_emb` and `support_emb` rows reproduce section 4 of `r9_analyse.py`.
2. AUC of the empty-graph control, base rate and median position of each group.
3. The seen - unseen difference within terciles of the position of first attempts (strata with at least 30 first attempts
   in each group, weighted by stratum size).
4. Spearman correlation of the first-attempt gain of `erR_emb` over `empty` with the update of the control's selected
   checkpoint (`best_step`), over units and within corpus.
5. Inventory of the run files of `runs_r8/` and `runs_r9/` by release, model, training seed and device (Supplementary
   Table S6.10).

Summaries as in `r9_analyse.py`: mean over corpora of subsample means, corpora with a positive value, unadjusted Wilcoxon p
over corpora, 95% mixed-model interval (random intercept per corpus) and corpus-then-subsample bootstrap interval.

| Step | Script | Output |
|---|---|---|
| Analysis | `src/r10_analyse.py` (`R9_RUNS`, `R8_RUNS` default to `runs_r9`, `runs_r8`) | `results_r10/r10_results.json` |
| Fig. 4 with the rung ER(R) - U | `src/r10_figures.py` (reads `results_r9/r9_results.json`, `results_r8/r8_results.json`) | `results_r10/fig4_ladder.png` |

In the manuscript that accompanies this release the ladder tables are Tables 6 (probe) and 7 (beyond the probe), the
robustness table is Table 8, the telescoping identity is Eq. (1) and the graph, the refinement and the attention logit are
Eqs. (2) to (4); the epoch-budget graph controls are Supplementary Table S5.2. Earlier sections of this file keep the
numbering of the manuscript at their release.

# Release 3.6.0: complete null-graph ladder, frozen random table, neighbour split, seed replicates

New runs for Sections 5.4, 6.4 to 6.6 and Supplementary Section S6 of the Neural Networks manuscript. All runs on a
CPU with one thread (`NT=1`), at 10% of the training learners under the update budget (at most 2000 updates, evaluation every
25, patience 8), so that they pair with the archived CPU runs of `runs_r6` and `runs_r7`; every rerun of an archived condition
reproduces its test AUC exactly (128 runs compared). 260 run files in `runs_r9/`, each with a
per-prediction file `<tag>_pred.npz` (p, y, new = first attempt, kc = target KC, nb_sup / nb_er = an in- or out-neighbour of
the target KC in the support / Erdos-Renyi graph occurs earlier in the learner's sequence, pos = position).

| Condition | Runs |
|---|---|
| `dp_emb` | 32 |
| `empty` | 32 |
| `empty` (seed replicate 1) | 8 |
| `empty` (seed replicate 2) | 8 |
| `erR_emb` | 32 |
| `er_emb` | 32 |
| `gkt_empty` | 4 |
| `gkt_er` | 4 |
| `gkt_rewired` | 4 |
| `gkt_rewired` (seed replicate 1) | 4 |
| `gkt_rewired` (seed replicate 2) | 4 |
| `gkt_support` | 4 |
| `gkt_support` (seed replicate 1) | 4 |
| `gkt_support` (seed replicate 2) | 4 |
| `gkt_uniform` | 4 |
| `support` | 32 |
| `support` (seed replicate 1) | 8 |
| `support` (seed replicate 2) | 8 |
| `support_emb` | 32 |

- `dp_emb`: degree-preserving rewiring read in the embedding refinement only; completes the ladder empty -> uniform -> ER -> DP
  -> support on 32 units of 10 corpora (`uniform_emb` from `runs_r7`).
- `erR_emb`: frozen-random-table control. The refinement mixes, over the Erdos-Renyi graph, a table R drawn once from N(0, 1)
  (seed 4000 + subsample) and never trained, instead of the learned table E; E, W_in, W_out are trained.
- `--rep j`: training seed sub + 100 j, same subsample and graphs (seed replicates on subsamples 0-1 of the four small corpora for the probe, of Statics 2011 and ASSISTments 2017 for GKT).
- `gkt_uniform`: GKT with the complete graph (Statics 2011, ASSISTments 2017, subsamples 0-1).

| Step | Script | Output |
|---|---|---|
| One run | `src/r9_controls.py --data <c> --sub <s> --cond <cond> [--rep j]` | `runs_r9/<tag>.json`, `_pred.npz` |
| Job list | `jobs_r9.txt` | |
| Analysis | `src/r9_analyse.py` | `results_r9/r9_results.json` |
| Figures 3 and 4 | `src/r9_figures.py` | `results_r9/fig3_learning_curve.png`, `fig4_ladder.png` |

Data: the ten corpora are rebuilt with `src/prep.py` (Junyi with `src/prep_junyi_lowmem.py`); the Junyi archive of the
mirror is RAR with a problem log over 2 GB, which this release extracted with the `unrar-cffi` Python package by streaming the
file to disk.

# Release 3.5.0: the null-graph ladder beyond the probe

New runs for Section 6.6 of the manuscript: update budget (at most 2000 updates, evaluation every 25,
patience 8), 10% of the training learners, six corpora (Statics 2011, Algebra 2005, Bridge to Algebra
and ASSISTments 2017 with subsamples 0-4; ASSISTments 2009 and 2015 with subsamples 0-1). 280 runs:

| Family | Conditions | Units |
|---|---|---|
| Probe (FKG-KT, graph in the embedding refinement only) | `empty`, `uniform_emb`, `er_emb`, `support_emb` | 24 |
| Parameter-sharing control | `hash`: each KC embedding is the mean of k rows of a shared table (k = clip(round(2E/K), 2, 8), rows drawn with seed 3000+s), no graph | 24 |
| SAKT-E (SAKT with the refined KC table) | `sakte_empty`, `sakte_uniform`, `sakte_er`, `sakte_support` | 24 |
| GKT (graph routes the state) | `gkt_empty`, `gkt_er`, `gkt_rewired`, `gkt_support` | 16 (subsamples 0-2 / 0-1) |

Graphs: support of the order-preserving graph of the subsample; Erdős–Rényi with as many edges (seed 2000+s);
degree-preserving rewiring of the support graph (seed 1000+s). GKT batch size 32/16/8 by number of KCs.

| Step | Script | Output |
|---|---|---|
| One run | `src/r8_controls.py --data <c> --frac 0.1 --sub <s> --cond <cond>` | `runs_r8/<tag>.json` |
| Job list | `jobs_r8.txt` | |
| All runs on Google Colab (resumable, writes to Drive) | `FKG_KT_r8_runs.ipynb` | |
| Analysis | `R8_RUNS=<dir> src/r8_analyse.py` | `results_r8/r8_results.json` |

The runs were made on a Colab T4 GPU, except the GKT runs on ASSISTments 2009 subsample 1 and on
ASSISTments 2015 (CPU); all conditions of one unit ran on the same device (field `device` of each run
file). Against the archived CPU runs of the same probe conditions (runs_r6, runs_r7) the GPU runs differ
by 0.0014 AUC on first attempts on average (at most 0.020; 96 runs), so the analysis pairs runs within
runs_r8 only. Per-prediction files (`_pred.npz`) were not archived for this release.
