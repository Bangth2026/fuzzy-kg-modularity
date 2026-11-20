# Binarisation and t-norm composition on fuzzy knowledge graphs

Reference implementation, reproduction scripts and run files for two manuscripts:

> **(A)** Hai Bang Truong. *Graded or crisp prerequisite knowledge? Auditing thresholding and t-norm path inference in fuzzy knowledge graphs built from learner logs.* Submitted to **Knowledge-Based Systems** (code: `fuzzykg/`, `scripts/`, `results/`; version 4.4.0 below).
>
> **(B)** Hai Bang Truong. *Where does a graph prior act inside a neural sequence model? Evidence from null-graph controls in knowledge tracing on ten corpora.* Submitted to **Neural Networks** (code: `kt/`; versions 2.0.0 to 3.9.0).

[![tests](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A fuzzy knowledge graph built from learner logs carries graded prerequisite edges. Pipelines that consume it discard the grades at two points: they threshold the graph before running a crisp community detector, and they replace max-min composition along paths by a non-idempotent t-norm. Manuscript (A) measures what each decision costs and turns the measurements into a pre-deployment audit (Algorithm 2, `fuzzykg/audit.py`, `scripts/fkg_audit.py`).

## Quick start: audit your own graph (Algorithm 2)

```bash
pip install -e .
# edges.csv: src,dst,mu (node labels or integer ids); optional grouping JSON {label: group};
# optional pairs.csv: pre,post,score (external prerequisite scores, e.g. expert ratings)
python scripts/fkg_audit.py edges.csv --nodes nodes.csv --grouping groups.json --pairs pairs.csv --out my_audit
```

The tool writes `my_audit.json` and a Markdown report: graph description; Algorithm 1 (degree-preserving null with the density-matched and the structure-matched floor, weights-only nulls with power); the Q_F drop and the rank-coded Q_F drop of every candidate cut over ten optimiser seeds, read against the no-cut baseline (a cut is flagged when it exceeds that baseline by more than twice the larger seed SD); agreement of every partition with the grouping; and, with external pairs, best paths for max-min, product, Łukasiewicz and Yager (p = 5) next to the direct edge and the hop distance, with a two-way cluster bootstrap over items. Example: `results/v4/audits/junyi_transition.md`, produced with `python scripts/make_junyi_pairs.py pairs.csv` and `python scripts/fkg_audit.py data/graphs/junyi_transition_edges.csv --nodes data/graphs/junyi_transition_nodes.csv --pairs pairs.csv --no-null-test --out results/v4/audits/junyi_transition`.

## What is in version 4.4.0

Analyses for the third revision of manuscript (A); new results in `results/v7/`. No earlier result changes.

| Topic | Script | Output |
|---|---|---|
| Ten-graph decision floor for variants of the Junyi transition construction near their floor | `scripts/v7_variant_floor.py` | `results/v7/variant_floor.json` (Supplementary Table S6d) |
| Seed dispersion of the Q_F drop on the real graphs, behind the flagging rule of Algorithm 2 | `scripts/v7_qf_drop_seed_sd.py` | `results/v7/qf_drop_seed_sd.json` (Table S3i) |
| Nested weighted SBM with five initialisations per fit; clipping of the logit covariate (option `SBM_CLIP`) on MOOCCubeX | `scripts/v5_weighted_sbm.py` | `results/v7/weighted_sbm_nested_fair.json`, `results/v7/weighted_sbm_clip*.json` (Tables S11b, S11c) |
| Fig. 1 with the weighted SBM and Algorithm 2 | `scripts/figure1.py` | `docs/figure1.png`, `docs/figure1.pdf` |

**Correction.** In the nested runs of `scripts/v5_weighted_sbm.py` (`SBM_NESTED=1`; `results/v6/weighted_sbm_nested.json`, `results/v7/weighted_sbm_nested_fair.json`) the membership covariate does not enter the description length (with and without it the description lengths agree to within 5 nats), so those runs carry no information on the weight channel and are not used; release 4.3.0 described them as finding no weight channel. The flat runs are not affected.

## What is in version 4.3.0

Analyses for the second revision of manuscript (A), the version on thirteen graphs from four corpora (MOOCCubeX, ASSISTments 2009/2015/2017, Junyi) whose numbering `docs/paper_map.md` follows. No earlier result changes; new results are in `results/v6/`.

| Topic | Script | Output |
|---|---|---|
| Junyi transition construction varied for the community question: conjunction (min, mean, product) x pruning threshold (0.1, 0.2, 0.3) x successor preference (max or sum denominator); Algorithm 1 in full, topic test, expert correlation | `scripts/v6_transition_counts.py`, `scripts/v6_transition_sensitivity.py` | `results/v6/transition_sensitivity*.json` (Supplementary Table S6c) |
| Decision floor of Algorithm 1 with ten planted graphs per in-community fraction and a bootstrap interval that re-reads the threshold | `scripts/v6_floor_uncertainty.py` | `results/v6/floor_uncertainty.json` (Table 3, Table S3h) |
| Weighted SBM with the same number of initialisations for the observed fit and every null refit (release 4.2.0 gave null refits one); nested SBM on five graphs | `scripts/v5_weighted_sbm.py` with `SBM_NULL_INIT=5 SBM_OUT=...` | `results/v6/weighted_sbm_fair.json`, `results/v6/weighted_sbm_nested.json`, `results/v6/sbm_partitions/` |
| Where the weight channel acts: inside-versus-between and directional differences of logit memberships with the SBM blocks fixed, against row permutations | `scripts/v6_sbm_where.py` | `results/v6/sbm_where.json` (Table S11a) |
| Finer tilt grid for the power of the weights-only nulls | `scripts/v6_power_fine.py` | `results/v6/power_fine.json` (Table 3) |
| Claim 1 check counted over 390 runs on the thirteen graphs | `scripts/v6_connectivity.py` | `results/v6/connectivity.json` |
| Complete pass over the MOOCCubeX exercise log (22.45 GB, HTTP range requests) and the concept-level counts the constructions need | `scripts/v6_mooccubex_scan.py`, `scripts/v6_mooccubex_log.py` | `results/v6/mooccubex_log.json` (Supplementary Section S12) |

The cross-fitted pruning threshold, the orientation bootstrap and the edges of the mean conjunction quoted in the revision come from release 4.2.0 (`results/v5/junyi_followup.json`, `results/v5/conjunction.json`).

## What is in version 4.2.0

Analyses added for the revision of manuscript (A). Every earlier result is unchanged; new results are in `results/v5/`, and `docs/figure2.png` is Fig. 2 of the main article. The main article now has seven tables (data, protocol, benchmarks, Algorithm 1 with the weighted SBM, the cut audit against groupings, composition on Junyi, decisions) and two figures; `docs/paper_map.md` maps each to its script and result file.

| Topic | Script | Output |
|---|---|---|
| Agreement of the cut audit with ARI loss: 2 × 2 tables, sensitivity, specificity, precision, balanced accuracy, Cohen's κ, Fisher exact p, bootstrap over graphs; flag-all and retained-share baselines | `scripts/v5_audit_agreement.py` | `results/v5/audit_agreement.json` |
| Weighted stochastic block model check of the membership question (degree-corrected SBM with logit(μ) as a real-normal edge covariate, row and global nulls refitted 20 times; inside/between and directional diagnostics) | `scripts/v5_weighted_sbm.py` (needs graph-tool, conda-forge) | `results/v5/weighted_sbm.json`, `results/v5/sbm_partitions/` |
| Sensitivity of the cut audit to the resolution γ ∈ {0.5, 1, 2} | `scripts/v5_gamma_audit.py` | `results/v5/gamma_audit.json` |
| Junyi: orientation of knowledge-map links by membership, precedence and gain, with a learner bootstrap; pruning threshold for the hop distance cross-fitted over exercises; correlation over pairs reached by every score at each threshold; learners who never accepted a suggestion | `scripts/v5_junyi_followup.py` | `results/v5/junyi_followup.json` |
| Junyi: which edges a compensatory conjunction (mean of asym and pref) adds, and their memberships within and between topics | `scripts/v5_conjunction.py` | `results/v5/conjunction.json` |
| Fig. 2 (correlation against the pruning threshold; cost of quantile cuts by construction) | `scripts/v5_figure2.py` | `docs/figure2.png`, `docs/figure2.pdf` |

`data/graphs/junyi_topics.json` maps every Junyi exercise to its topic (from `junyi_Exercise_table.csv`), so that the Junyi groupings no longer need the raw files. Note for anyone reusing graph-tool: edge covariates need `WeightedBlockState` (`minimize_blockmodel_dl(g, state=WeightedBlockState, state_args=dict(rec=[w], rec_types=[...]))`); passing `recs` to `BlockState` is accepted without error and ignored.

## What is in version 4.1.0

The computations and results are those of 4.0.0, and no number changed. The release matches the version of manuscript (A) shortened to the length that Knowledge-Based Systems prefers: the main article keeps Lemma 1, Proposition 1, Algorithms 1 and 2 and four tables, and Supplementary material B (an extended report) holds the complete derivations and per-graph results. `docs/paper_map.md` maps every item of the main article to the corresponding item of Supplementary material B, the script that computes it and its result file. Comments in the code follow the numbering of Supplementary material B (for example, Lemma 2 there is Lemma 1 of the main article).

## What is in version 4.0.0 (revision for Knowledge-Based Systems)

All new results are in `results/v4/`; `scripts/v4_summarise.py` assembles the tables of the revised manuscript (`results/v4/tables.md`) and every quoted number (`results/v4/paper_numbers.json`).

| Topic | Script | Output |
|---|---|---|
| Five further corpora (ASSISTments 2012, KDD Cup 2010 Algebra 2005 and Bridge to Algebra 2006, Eedi, EdNet), both constructions, external groupings | `scripts/v4_build_new.py`, loaders in `data/loaders.py` | `data/graphs/<corpus>_*`, `results/v4/new_corpora_stats.json` |
| Descriptive statistics of every log after preprocessing | `scripts/v4_corpus_stats.py` | `results/v4/corpus_stats.json` |
| Algorithm 1 on the ten new graphs | `scripts/v4_algorithm1.py --new` | `results/v4/algorithm1_new.json` |
| Structure-matched (degree-corrected) decision floor on all 23 graphs | `fuzzykg.nulltest.structure_matched_floor`, `scripts/v4_algorithm1.py --dcfloor` | `results/v4/dcfloor.json` |
| Cost decomposition and Table 4 rows for the new graphs | `scripts/cost_decomposition.py`, `scripts/v4_table4_new.py` | `results/v4/cost_decomposition_new.json`, `results/v4/table4_new.json` |
| Cut audit against external groupings, observed and tilted memberships | `scripts/v4_protocol_external.py` | `results/v4/protocol_external.json` |
| Sensitivity to horizon, support, pruning threshold and membership operator | `scripts/v4_sensitivity.py` | `results/v4/sensitivity.json` |
| Next concept with popularity, recency, bigram, second-order Markov and GRU baselines; Junyi without suggested attempts | `scripts/v4_nextconcept_baselines.py` | `results/v4/nextconcept_*.json` |
| Inter-rater reliability of the Junyi crowd labels | `scripts/v4_junyi_labels.py` | `results/v4/junyi_labels.json` |
| Negative single-seed cost indices and seed sensitivity | inline in `scripts/v4_summarise.py` | `results/v4/index_negativity_seeds.json` |
| Run time and peak memory against nodes and edges | `scripts/v4_scaling.py` | `results/v4/scaling.json` |
| Algorithm 2 as a tool | `fuzzykg/audit.py`, `scripts/fkg_audit.py`, `tests/test_v4.py` | `results/v4/audits/` |

Code changes: `fuzzykg/nulltest.py` adds `planted_dc_like`, `structure_matched_floor` and `outcome`; `fuzzykg/fastcount.encode` accepts a categorical KC column with the same coding as strings (the Junyi log now loads in about 4 GB); `data/loaders.py` adds loaders for the five corpora, a parquet cache and `load_any`. Every graph of release 1.4.3 is rebuilt byte for byte (checked for the ASSISTments 2017 and Junyi transition graphs).

## What is in version 3.9.0

No new training runs. `kt/src/r12_analyse.py` (`kt/results_r12/`) splits the first-attempt contrasts of the random-vector
controls by training interactions on the target KC (n_k = 0 against n_k > 0, and pooled quintiles of log n_k with a slope
per decade), after checking the rebuilt corpora against the archived per-prediction files, and lists every run set of
Section 5.4 of the manuscript (721 runs). It also explains the names of the run and result directories. See the section
"Release 3.9.0" of `kt/README.md`.

## What is in version 3.8.0

The random-vector controls with the two maps of the embedding refinement frozen at initialisation (`rvecF_emb`, `rvec1F_emb`;
64 CPU runs, `kt/src/r11_controls.py`, `kt/runs_r11/`), the support graph on the rebuilt EdNet data, and reruns that reproduce
the archived control exactly. `kt/src/r11_analyse.py` (`kt/results_r11/`) adds the support graph against the controls without a
graph, the frozen-map comparisons, first attempts at KCs with and without training interactions, the share of KCs without an
Erdős–Rényi neighbour, and the units and strata behind the neighbour split by position. See the section "Release 3.8.0" of
`kt/README.md`.

## What is in version 3.7.0

Controls without a graph: a fixed random vector per KC with no graph (two variances) and Erdős–Rényi mixing of a second, trainable table, on the 32 units of the complete ladder (96 CPU runs, `kt/src/r10_controls.py`, `kt/runs_r10/`), on corpora rebuilt from public copies of the raw files, with reruns that reproduce the archived control exactly on nine corpora. Analyses of the per-prediction files: first attempts split by a previously practised neighbour for the frozen-random-table control and for the learned-minus-frozen contrast, the same split within terciles of the position in the sequence, the AUC of the empty-graph control in each group, training length, an inventory of the runs of Section 5.4, and Fig. 4 with the rungs ER(R) − U and RV − U (`kt/src/r10_analyse.py`, `kt/src/r10_figures.py`, `kt/results_r10/`). See the section "Release 3.7.0" of `kt/README.md`.

## What is in version 3.6.0

The complete null-graph ladder with the graph read in the embedding refinement (32 units of 10 corpora), the frozen-random-table control, first attempts split by a previously practised graph neighbour, seed replicates for the probe and for GKT, GKT with the complete graph, and Figures 3 and 4 of the manuscript: 260 CPU runs with per-prediction files (`kt/runs_r9/`), `kt/src/r9_controls.py`, `kt/src/r9_analyse.py`, `kt/src/r9_figures.py`, results in `kt/results_r9/`. See the section "Release 3.6.0" of `kt/README.md`.

## What is in version 3.5.0

The null-graph ladder beyond the probe (Section 6.6 of the manuscript): 280 runs on six corpora at 10% of the training learners under the update budget. GKT with the empty, Erdős–Rényi, degree-preserving and support graphs; SAKT with the embedding refinement over the ladder empty, uniform, Erdős–Rényi, support; a parameter-sharing (hashed) KC table; and the paired probe runs. Code in `kt/src/r8_controls.py` and `kt/src/r8_analyse.py`, run files in `kt/runs_r8/`, results in `kt/results_r8/r8_results.json`, the Colab notebook that ran them in `kt/FKG_KT_r8_runs.ipynb`. See the section "Release 3.5.0" of `kt/README.md`.

## What is in version 3.4.0

Robustness of the main comparisons and decision-level metrics, and the uniform-mixing, tuned-shrinkage and learning-rate controls (`kt/src/r34_analyse.py`, `kt/src/r7_controls.py`, `kt/runs_r7/`). See the section "Release 3.4.0" of `kt/README.md`. Archived on Zenodo: [10.5281/zenodo.24131158](https://doi.org/10.5281/zenodo.24131158).

## What is in version 3.3.0

Channel-separating controls (`support_emb`, `support_att`, `er_emb`, `er_att` in `kt/src/r6_controls.py`, 128 run files in `kt/runs_r6`) and `kt/src/r33_analyse.py` with `kt/results_r6/r33_results.json`: the graph read only in the embedding refinement or only in the attention bias; rise of the graph gain from 5% to 10% under both budgets on the same units; updates to the selected checkpoint; graph controls pooled over both budgets with a budget fixed effect. See the section "Release 3.3.0" of `kt/README.md`.

## What is in version 3.2.0

`kt/src/r32_analyse.py`, new run files in `kt/runs_r6/` and `kt/results_r6/r32_results.json`: the degree-feature control with a zero-initialised map (`degfeat0`), under the epoch and the update budget; the Erdős–Rényi control on every subsample of the epoch-budget control runs and on all ten corpora under the update budget; the degree-preserving control on all subsamples under the update budget; pooled comparisons of the graph controls over both budgets with intervals; run counts by set. See the section "Release 3.2.0" of `kt/README.md`.

## What is in version 3.1.0

`kt/src/r6_controls.py`, `kt/src/r6_analyse.py`, run files in `kt/runs_r6/` and results in `kt/results_r6/`: the probe trained for a fixed number of parameter updates instead of 30 epochs (support graph, empty-graph control, degree-preserving rewiring and an Erdős–Rényi graph with the same number of edges, at 5% and 10% of the training learners); two further graph controls under the epoch budget (Erdős–Rényi graph, and the empty graph with a degree feature in the KC embedding); the number of updates taken by every learning-curve run; the support-versus-control comparison pooled over the learning-curve runs and the runs of version 2.1.0; bootstrap and mixed-model intervals for the controls; and run counts by set. See the section "Release 3.1.0" of `kt/README.md`.

## What is in version 3.0.0

`kt/r2/`: code, the 957 learning-curve run files and their analysis (Sections 6.1 to 6.5 and 6.9 of the manuscript); see `kt/r2/README.md`.

## What is in version 2.1.0

Archived on Zenodo: [10.5281/zenodo.24027065](https://doi.org/10.5281/zenodo.24027065) (all versions: 10.5281/zenodo.23830894).

The knowledge-tracing experiments live in `kt/` (see `kt/README.md`). Version 2.1.0 adds graph controls: a graph-free recency bias and a degree-preserving rewiring of the graph, run with the release-2.0.0 model on ten corpora (five subsamples at 10%, three at 25%); the empty-graph control rerun on the 10% sample of release 2.0.0 with three seeds; the support graph against the control at 5%, 10%, 25% and 100%; one cell with a larger learning rate for the graph scalars; preprocessing of Bridge to Algebra 2006 and Statics 2011; and a memory-light Junyi preprocessing. Run files in `kt/runs_r5/`, results in `kt/results_r5/`.

## What is in version 2.0.0

`kt/`: preprocessing of eight corpora, fuzzy prerequisite graphs from training learners, DKT, SAKT, AKT, GKT and FKG-KT training on CPU, and the 527 run files of release 2.0.0 (archived on Zenodo, 10.5281/zenodo.23830895).

## What is in version 1.4.3

- The rank-coded Q_F drop implied by the level-set identity: one Leiden run on memberships replaced by their quantile-grid ranks; validity against the recovery shortfall, disagreements with the cost index and the Q_F drop, and the size of the covariance term (`scripts/rank_drop_audit.py`, `scripts/rank_drop_summary.py`; merged into `scripts/cost_index_validity_extended.py`).
- Junyi transition graph rebuilt at five pruning thresholds (none, 0.05, 0.1, 0.2, 0.3) and evaluated on one fixed set of annotated pairs with cluster-bootstrap intervals for Spearman rho and Kendall tau-b (`scripts/junyi_pruning_fixed_pairs.py`); graded composition without pruning against hop count at every threshold on the same resamples (`scripts/junyi_pruning_cross.py`).
- Degree-preserving null with more draws for one graph (`scripts/degree_null_more_draws.py`).
- Equal-size binarisations on the LFR-style benchmark (`scripts/nonnested_lfr.py`).

## What is in version 1.4.2

- Exact finite identity between the level-set (multislice) modularity on any grid and the fuzzy-graph modularity of transformed memberships minus a covariance of degree shares across levels, with a numerical check (`scripts/check_levelset_identity.py`).
- Binarisations of equal size by four rules (global threshold, per-source top share, disparity filter, random thinning) on the planted sweep, and partial correlations of the cost index and the Q_F drop with the recovery shortfall given retained mass and mean out-degree (`scripts/nonnested_binarisation.py`, `scripts/nonnested_summary.py`).
- Decisions under rescaling: pairs of cuts whose order changes under monotone rescalings (`scripts/rescaling_decisions.py`); cost index and Q_F drop on co-practice graphs rebuilt at three orness values, quantile and absolute cuts (`scripts/orness_decision.py`).
- Junyi expert criterion with a two-way cluster bootstrap over exercises, Kendall tau-b, ranking over reached pairs, and transition graphs rebuilt at pruning thresholds 0.1 and 0.3 (`scripts/junyi_cluster_bootstrap.py`).
- Held-out next-concept MRR under three tie rules (expected reciprocal rank under random tie-breaking, average rank, pessimistic rank) (`scripts/heldout_transitions.py`).
- Weights-only nulls with more draws for one graph (`scripts/weights_null_more_draws.py`); edge-level agreement with the Junyi expert scores with cluster-bootstrap intervals (`scripts/junyi_expert_cluster.py`).

## What is in version 1.4.1

- Cost decomposition on the real graphs: no-cut (support) baseline, optimiser-seed dispersion of every index, and the index under permuted memberships (`scripts/cost_decomposition.py`); the Q_F drop of the same cuts (`scripts/real_qf_drop.py`).
- Validity of the index against the recovery shortfall 1 − ARI(cut, truth), within-graph concordance, bootstrap intervals, and the Q_F drop, relative and equal-weighted indices as competitors (`scripts/cost_index_validity_extended.py`); dependence on the unit of μ_E under monotone rescalings (`scripts/rescaling_validity.py`).
- Memberships against groupings fixed independently of the graph (Junyi topics and areas, ASSISTments skills and assignments), with global and within-source permutations (`scripts/external_reference_test.py`), and the topic agreement of Leiden on Q_F against permuted memberships (`scripts/junyi_topic_partition_check.py`).
- Composition beyond reachability: an unweighted hop-distance baseline, common-support correlations, split-half operator selection and the density mechanism (`scripts/composition_robustness.py`, `scripts/hop_vs_tnorm.py`); held-out prediction with the hop baseline and the operator selected on validation learners (`scripts/heldout_transitions.py`).
- Sensitivity of the reference and the disparity filter on all thirteen graphs (`scripts/levelset_sensitivity.py --part real`).

## What is in version 1.4

- Junyi Academy as a fourth corpus: loader for the problem log and for the expert information (exercise table with knowledge-map parents, topics and areas; pairwise crowd annotations), both constructions, and a check that rebuilds the transition graph without the attempts the platform itself suggested.
- The order-preserving transition construction (`build_transition_graph`), run on every log.
- Compiled counting kernels (numba) for both constructions, byte-identical to the reference implementation (`fuzzykg/fastcount.py`, tested).
- The calibrated null test (`fuzzykg/nulltest.py`, `scripts/calibrated_null_test.py`).
- Fast all-pairs best paths for max-min and the Archimedean families, with hop limit and floor (`fuzzykg/paths_fast.py`, tested against the reference implementation).
- Expert-labelled criteria for composition (`scripts/prerequisite_criterion.py`, `scripts/junyi_expert_validity.py`) and held-out next-concept prediction with paired, Holm-adjusted comparisons (`scripts/heldout_transitions.py`).
- Validity of the cost index against a ground-truth recovery loss (`scripts/cost_index_validity.py`); sensitivity of both constructions to the horizon and support, and a bootstrap of the partitions over learners (`scripts/construction_sensitivity.py`).

| Topic | Where | Output |
|---|---|---|
| Graph statistics and partitions of the 13 real graphs | `scripts/real_graph_tables.py` | `results/real_graph_tables.json` |
| Multislice reference and cost index on real graphs | `scripts/levelset_real.py` | `results/levelset_real.json` |
| Planted sweep, LFR-style benchmark | `scripts/benchmark_levelset.py --stage shift / lfr` | `results/levelset_{shift,lfr}.json` |
| Fixed-information planted benchmark | `scripts/benchmark_binarisation.py --graph-seeds 5` | `results/binarisation.json` |
| Cost index against ground-truth loss | `scripts/cost_index_validity.py` | `results/cost_index_validity.json` |
| Calibrated null test | `scripts/calibrated_null_test.py` | `results/calibrated_null_test.json` |
| Junyi edge validity | `scripts/junyi_expert_validity.py` | `results/junyi_expert_validity.json` |
| Composition against expert labels | `scripts/prerequisite_criterion.py`, `scripts/junyi_unsuggested.py` | `results/prerequisite_criterion.json`, `results/junyi_unsuggested.json` |
| Held-out next concept | `scripts/heldout_transitions.py --release {junyi,2009,2015,2017} --level {kc,item} [--construction transition] [--max-hops L --tau-path t]` | `results/heldout_*.json` |
| Construction sensitivity, partition bootstrap | `scripts/construction_sensitivity.py --part grid / bootstrap` | `results/construction_sensitivity.json`, `results/membership_bootstrap.json` |
| Sensitivity of the reference (grid, seeds, disparity backbone) | `scripts/levelset_sensitivity.py` | `results/levelset_sensitivity_*.json` |
| Propositions, ordinal invariance | `scripts/verify_propositions.py`, `scripts/scale_invariance_check.py` | stdout, `results/scale_invariance.json` |

## Installation

```bash
git clone https://github.com/Bangth2026/fuzzy-kg-modularity.git
cd fuzzy-kg-modularity
pip install -e ".[dev]"          # numpy, scipy, pandas, scikit-learn, numba, python-igraph, leidenalg, pytest
pytest
```

Python ≥ 3.10. Everything runs on one CPU core. The results in the paper were produced with Python 3.13, python-igraph 1.0.0, leidenalg 0.12.0, numpy 2.5.3 and scipy 1.18.1. Without numba the counting falls back to the pure-Python path, which gives the same graphs more slowly.

## Data

Download instructions and expected paths are in [`data/README.md`](data/README.md). Nothing is redistributed here except the derived graphs in `data/graphs/`.

```bash
python scripts/build_mooccubex_graph.py [--full]
python scripts/build_assistments_graph.py --release 2009 --level kc [--construction transition]
python scripts/build_junyi_graph.py --construction both
```

## Package layout

```
fuzzykg/
  graph.py          FuzzyGraph: directed, μ_E ∈ (0,1], α-cuts, strengths, reciprocity
  modularity.py     Q_F, the α-average and its gap
  leiden.py         Leiden on Q_F via leidenalg (RBConfiguration, directed, weighted)
  multislice.py     multislice reference Qbar over a quantile grid of α-cuts, multiplex Leiden, cost index
  nulltest.py       calibrated null test: degree-preserving rewiring, density-matched floor, weights-only nulls, power by tilt
  tnorms.py         t-norms, Yager and Schweizer–Sklar families, generators, OWA
  paths_fast.py     all-pairs best-path strengths (widest path; Dijkstra on the generator; hop-limited products)
  construction.py   co-practice and transition constructions, preprocessing
  fastcount.py      compiled counting kernels for the constructions
  synthetic.py      planted partitions with membership shift, LFR-style directed benchmark, scalability generator
  backbone.py       disparity-filter backbone on a directed fuzzy graph
  fuzzification.py  interval-valued learner state (Supplementary material)
data/loaders.py     ASSISTments 2009/2015/2017, MOOCCubeX-CS and Junyi loaders with the stated preprocessing
scripts/            reproduction scripts
tests/              pytest tests
```

## Citation

```bibtex
@article{truong2026fuzzykg,
  author  = {Truong, Hai Bang},
  title   = {Graded or crisp prerequisite knowledge? Auditing thresholding and t-norm path inference in fuzzy knowledge graphs built from learner logs},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {submitted},
}
```

Archived on Zenodo: https://doi.org/10.5281/zenodo.24131157 (all versions; resolves to the latest). Version 3.9.0: https://doi.org/10.5281/zenodo.24212530. See also `CITATION.cff`.

## License

MIT for the code (see `LICENSE`). Datasets remain under their own licenses (Junyi Academy: non-commercial use, cite Chang, Hsu and Chen, EDM 2015); see `data/README.md`.

## Acknowledgements

The ASSISTments team at Worcester Polytechnic Institute, THU-KEG for MOOCCubeX, Junyi Academy and the authors of its public release, and the authors of `leidenalg` and `python-igraph`.
