# Binarisation and t-norm composition on fuzzy knowledge graphs

Reference implementation, reproduction scripts and run files for the manuscript:

> **(A)** Hai Bang Truong. *Where do the grades of a log-derived fuzzy prerequisite graph carry value? A null-model decomposition and a unit-robust audit across edge, path and partition use.* Submitted to **Knowledge-Based Systems** (code: `fuzzykg/`, `scripts/`, `results/`; versions 5.0.0 to 5.4.0 below; versions up to 4.11.0 accompanied the earlier form of the manuscript, *Should prerequisite knowledge be graded or crisp? Quantifying what thresholding and t-norm path inference cost in log-derived fuzzy knowledge graphs*).


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

The tool writes `my_audit.json` and a Markdown report: graph description; Algorithm 1 (degree-preserving null with the density-matched and the structure-matched floor, weights-only nulls with power); the Q_F drop and the rank-coded Q_F drop of every candidate cut over 20 optimiser seeds, read against the no-cut baseline (a cut is flagged when it exceeds that baseline by more than twice the larger seed SD and by more than 0.01; see "What is in version 4.10.0" for the decision by unit and step 4, and `--gamma` (version 4.11.0) for the resolution, 1 by default); agreement of every partition with the grouping; and, with external pairs, best paths for max-min, product, Łukasiewicz and Yager (p = 5) next to the direct edge and the hop distance, with a two-way cluster bootstrap over items. Example: `results/v4/audits/junyi_transition.md`, produced with `python scripts/make_junyi_pairs.py pairs.csv` and `python scripts/fkg_audit.py data/graphs/junyi_transition_edges.csv --nodes data/graphs/junyi_transition_nodes.csv --pairs pairs.csv --no-null-test --out results/v4/audits/junyi_transition`.

## What is in version 5.4.2

The directory `kt/` (a separate knowledge-tracing study) and its test file are removed; that code now lives in
https://github.com/Bangth2026/kt-null-graph-controls. No code or result of manuscript (A) changes.

## Relation to ASOC-D-26-17250 and shared data (release 5.4.1)

A related single-authored manuscript, *Traceable Construction and Abstention for Graded Prerequisite Graphs in Tutoring Systems* (Applied Soft Computing, ASOC-D-26-17250, under review; code and data archive https://doi.org/10.5281/zenodo.23074830), uses data that overlap with this repository:

| Item | ASOC-D-26-17250 | This repository (manuscript A) |
|---|---|---|
| Junyi Academy log (PSLC DataShop 1198) | used | used |
| Junyi crowd-annotated exercise pairs | 1,131 pairs, plus 823 held out | all 1,954 pairs, as the path criterion |
| Within-learner adjacent transitions (about 25.7 million) | undirected co-occurrence criterion | the counts behind the Junyi transition graph (`scripts/build_junyi_graph.py`) |
| Graded crowd weights vs majority-vote weights | main result (Q1) | **not computed, not reported** |
| ASSISTments 2009-2010 | second-platform directed check | KC- and item-level graphs |
| Pruning threshold audit, unit-robust cut audit, t-norm and hop-penalised path scores | not addressed | contributions |

Manuscript A cites ASOC-D-26-17250 as the source of the adjacent-transition definition and of the finding that graded weights on the Junyi data carry information binary ones lose; its Junyi results are read as an application of the threshold and path-score analysis, not as independent evidence on that question. Release 5.4.1 changes no code or result file.

## What is in version 5.4.0

The analyses of the fourth revision of manuscript (A); new results in `results/v19/`. No earlier result changes, except the Table S9c cross-fit, which is recomputed by `scripts/v19_misc.py` (its earlier values had no script in the repository). `REPRODUCE.md` maps every main-text table and figure to its script, result file and run time; `requirements-lock.txt` pins the environment.

| Topic | Script | Output |
|---|---|---|
| Three-level reach score, threshold chosen on held-out exercises, Table S9c cross-fit | `scripts/v19_misc.py` | `results/v19/misc.json` |
| Partition value at 64 interior points of B_2, null per seed triple | `scripts/v19_interior.py` | `results/v19/interior.json` |
| Guide in use: prerequisite and next-concept recommendation; bootstrap p-values | `scripts/v19_kbs.py` | `results/v19/kbs.json` |
| Permutation tests, noise, scale, graph features | `scripts/v19_robust.py` | `results/v19/robust.json` |

## What is in version 5.3.0

The analyses of the third revision of manuscript (A); new results in `results/v18/`, every new number in `results/v18/summary.json`. No earlier result changes.

| Topic | Script | Output |
|---|---|---|
| Reach and order of the path correlation over the pruning threshold (Table 4 of the manuscript, Table S9f) | `scripts/v18_reach.py` | `results/v18/reach.json` |
| Cross-fitted best crisp vs best graded with the scores of Tables S4b and S9c (Table S4g) | `scripts/v18_crossfit.py` | `results/v18/crossfit.json` |
| Partition value over all 512 vertices of B_2 and six seed triples (Table S4i); Junyi with and without suggested attempts on a common vertex set, edges only with suggested attempts (Table S4h) | `scripts/v18_partition.py` (`unsuggested`, `box`) | `results/v18/partition.json` |
| Summary | `scripts/v18_summarise.py` | `results/v18/summary.json` |

## What is in version 5.2.0

The analyses of the second revision of manuscript (A); new results in `results/v17/`, every new number in `results/v17/summary.json`, every new or replaced supplementary table in `results/v17/supp_tables.json`; figures `docs/v52_figure*` (Figures 3 and 4 swap numbers with respect to 5.1.0). `fuzzykg.simplex` now enumerates the box vertices up to J = 12, so the refined family of Corollary 1 is searched like the family (5); no result of the family (5) changes (J = 9).

| Topic | Script | Output |
|---|---|---|
| Divergence check with a null reference (row permutations), on the injected worlds and the real graphs | `scripts/v17_divergence_null.py` (`PART=0/2`, `PART=1/2`) | `results/v17/divergence_null_part*.json` |
| Best crisp vs best graded over the pruning threshold (cross-fitted); grouping-class null of the partition value, Junyi without suggested attempts; sparse minimum excess 0.03 | `scripts/v17_misc.py` | `results/v17/misc.json` |
| kappa* in the refined family with full vertex enumeration | `scripts/v17_refined.py` | `results/v17/refined.json` |
| kappa* of the Junyi transition cuts against the share of learners (needs the Junyi log) | `scripts/v17_junyi_fraction.py` | `results/v17/junyi_fraction.json` |
| Checks calibrated by density class; summary; supplementary tables | `scripts/v17_summarise.py`, `scripts/v17_supp_tables.py` | `results/v17/summary.json`, `results/v17/supp_tables.json` |

## What is in version 5.1.0

The analyses of the revised manuscript (A). New code in `fuzzykg/simplex.py` (`refined_knots`, `divergence`), `fuzzykg/semisynthetic.py` (`World.delta`), new results in `results/v16/`, figures in `docs/v51_figure*.{png,pdf}`; `docs/paper_map_r11.md` maps every new table and figure to its script. No earlier result file changes. Every new number of the manuscript is in `results/v16/summary.json`, every new supplementary table in `results/v16/supp_tables.json`.

| Topic | Script | Output |
|---|---|---|
| Value of grades of the Junyi transition graph at pruning thresholds 0, 0.05, 0.1, 0.2, 0.3 on the fixed 1,954 annotated pairs (needs the Junyi problem log) | `scripts/v16_junyi_pruning.py` | `results/v16/junyi_pruning_value.json`, `data/graphs/junyi_transition_tau*` |
| Path and partition values over the 18 vertices of the box B_2 of rescalings | `scripts/v16_units.py` | `results/v16/units.json` |
| Real graphs from the cached partitions: losses against Pi_F, row-null gain check, refined family, kappa* over seed resamples, seed-triple ranges and Monte Carlo errors of the partition value, column null | `scripts/v16_real.py` | `results/v16/real.json` |
| Semi-synthetic worlds with community information in the generator, delta in {0, 0.2, 0.4} | `scripts/v16_semisynthetic.py` | `results/v16/semisynthetic_delta_*.json` |
| Grades made to mark the communities after construction, eta in {0, 0.5, 1, 2}; divergence check | `scripts/v16_injected.py`, `scripts/v16_divergence.py` | `results/v16/injected_part*.json`, `results/v16/divergence.json` |
| Learner subsamples without duplicates (ASSISTments KC level, Junyi transition) and the bootstrap on Junyi | `scripts/v16_learner_resampling.py` | `results/v16/learner_resampling.json` |
| All new numbers; supplementary tables; Figures 1, 2 and 4 | `scripts/v16_summarise.py`, `scripts/v16_supp_tables.py`, `scripts/v16_figures.py` | `results/v16/summary.json`, `results/v16/supp_tables.json`, `docs/v51_figure*` |

Run order: the v15 run order, then `v16_junyi_pruning.py`, `v16_units.py`, `v16_real.py`, `v16_semisynthetic.py 0 0.2 0.4`, `v16_injected.py` (optionally split with `PART=0/2` and `PART=1/2`), `v16_divergence.py`, `v16_learner_resampling.py`, `v16_summarise.py`, `v16_supp_tables.py`, `v16_figures.py`. The ASSISTments 2009 loader reads `skill_builder_data.csv`; the `_corrected` release of the EduData mirror reproduces the graphs of `data/graphs/` and can be linked under that name.

## What is in version 5.0.0

The analyses of the restructured manuscript (A). New code in `fuzzykg/simplex.py`, `fuzzykg/pathscore.py` and `fuzzykg/semisynthetic.py`, new results in `results/v15/`, figures in `docs/v5_figure*.{png,pdf}`, and `docs/paper_map_r10.md` maps every table and figure to its script. No earlier result file changes. Every number of the manuscript is in `results/v15/summary.json`.

| Topic | Script | Output |
|---|---|---|
| Leiden partitions of the 23 educational graphs (support, quantile cuts, Q_F; nine resolutions; global and row permutations) reused by every v15 script | `scripts/v15_partitions.py` | `results/v15/partitions/*.npz` |
| Unit-robust cut audit (Algorithm 1): excess of a cut in closed form over a cone of increasing rescalings (Proposition 1), its range over the family (Proposition 2), robustness radius kappa*, resolution sweep, minimum excess 0.0025 to 0.05, external groupings | `scripts/v15_simplex_real.py` | `results/v15/simplex_real.json` |
| Semi-synthetic learner logs from a known prerequisite DAG with known communities (96 worlds, 192 graphs): audit against the true loss, calibration of the minimum excess, paths, value of grades | `scripts/v15_semisynthetic.py` | `results/v15/semisynthetic.json` |
| Value of grades (edge, path, partition) with global and row nulls on the Junyi graphs and the graphs with a grouping; connectivity level of max-min (Proposition 3); hop-penalised score with cross-fitted lambda | `scripts/v15_value_paths.py` | `results/v15/value_paths.json` |
| The same path analyses on PPI5k and CN15k | `scripts/v15_crossdomain.py` | `results/v15/crossdomain.json` |
| Stability of the decisions under a bootstrap of the learners (ASSISTments KC level, both constructions) | `scripts/v15_learner_bootstrap.py` | `results/v15/learner_bootstrap.json` |
| Numerical check of Proposition 1, check of the kappa* search, cost on Junyi co-practice | `scripts/v15_checks.py`, `tests/test_simplex.py` | `results/v15/checks.json` |
| All numbers of the manuscript; Figures 1 to 3 | `scripts/v15_summarise.py`, `scripts/v15_figures.py` | `results/v15/summary.json`, `docs/v5_figure*` |

Run order: `v15_partitions.py`, `v15_simplex_real.py`, `v15_semisynthetic.py`, `v15_value_paths.py`, `v15_crossdomain.py`, `v15_summarise.py` (selects the minimum excess by density class on the calibration worlds), `v15_learner_bootstrap.py` and `v15_checks.py` (both read the selected values), `v15_summarise.py` again, `v15_figures.py`. Files ending in `_pass1.json` and `learner_bootstrap_delta0.01.json` are the first passes, run before the minimum excess was calibrated, kept for transparency. The Junyi problem log is not needed by any v15 script (the graphs are in `data/graphs/`); the Junyi exercise table and annotation files are.

## What is in version 4.11.0

Changes that answer the ninth review of the Knowledge-Based Systems version of manuscript (A); new results in `results/v14/`, and `docs/paper_map_r9.md` maps each item to the manuscript. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Resolution of step 3: the step re-run on mu_E of the thirteen real graphs at gamma 0.5, 1 and 2 (every Leiden run and every Q_F of the drops at that gamma; 20 seeds, absolute reading and unit rule). Gamma 1 reproduces all 39 decisions of `results/v12/unit_rule.json`; gamma 0.5 changes 9 and gamma 2 changes 8; sparse lower-quartile cuts flagged 4, 4 and 6 of 7; median number of communities of Pi_F per gamma | `scripts/v14_gamma_step3.py` (`--communities` adds the counts) | `results/v14/gamma_step3.json` (Section 8.5, Supplementary Section S3) |

`fuzzykg.audit.cut_audit` and `audit` take `gamma` (default 1.0) and `scripts/fkg_audit.py` takes `--gamma`; the manuscript decides at 1 and recommends running step 3 at the resolution at which the partition will be used.

## What is in version 4.10.0

Changes that answer the eighth review of the Knowledge-Based Systems version of manuscript (A); new results in `results/v13/`, and `docs/paper_map_r8.md` maps each item to the manuscript. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Absolute reading of step 3 where the no-cut drop exceeds 0.05 (only under x^3): a cut is flagged when either the relative or the absolute reading flags it. Sparse lower-quartile counts under every rescaling and the Junyi topic check redone with this reading (x^3 now flags 5 of 7 and passes the check; only the rank-coded drop is rejected; the range stays 4 to 7) | `scripts/v13_absolute_reading.py` | `results/v13/absolute_reading.json` (Section 8.3, Supplementary Section S3) |
| Permutation reference of step 4: ARI of the best of three seeds of G_F with the external grouping against 20 row-wise and 20 global permutations of mu_E. Junyi transition: 0.685, above every permutation; Junyi co-practice: 0.060, below the permutation mean, so that grouping cannot show that an undecided cut loses nothing | same | same |

`fuzzykg.audit.cut_audit` and `audit` take `perm_draws` (default 20). With a grouping, they report `grouping_reference` and, for every undecided cut, a `step4` entry: the cut becomes `cut (step 4: no loss)` only when the grouping is matched above the 95th percentile of its permutation reference and the cut's ARI is within 2 seed SDs of that of Pi_F; with an informative grouping and a loss it becomes `flagged (step 4: loss)`; otherwise it stays `undecided`. Each flag is reported as `flag_*_relative` and, on graphs with a baseline warning, `flag_*_absolute`. The threshold 0.05 was set in release 4.9.0 after the drops under x^3 had been seen; no decision on mu_E depends on it.

## What is in version 4.9.0

Changes that answer the seventh review of the Knowledge-Based Systems version of manuscript (A); new results in `results/v12/`, and `docs/paper_map_r7.md` maps each item to the manuscript. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Step 3 of Algorithm 2 read by the unit of the memberships: where the unit is meaningful (count shares, released or calibrated scores) the Q_F drop on mu_E decides; where it is arbitrary a cut is cut only when the Q_F drop and the rank-coded drop both pass, and "undecided" when they disagree. Counts by construction, lower-quartile counts on the sparse graphs under every rescaling, the range under the rescalings the Junyi topic check does not reject, cells near a threshold, no-cut drops above 0.05 | `scripts/v12_unit_rule.py` | `results/v12/unit_rule.json` (Supplementary Section S3, Table 4, Table 6) |
| Wall-clock cost of step 3 as Algorithm 2 specifies it (20 seeds, Q_F and rank-coded drops) on Junyi co-practice, with ten seeds in the same session for comparison | `scripts/v12_step3_cost.py` | `results/v12/step3_cost_junyi_copractice.json` (Table S15) |
| Function index of every script | - | `scripts/INDEX.md` |

`fuzzykg.audit.cut_audit`, `audit` and `scripts/fkg_audit.py` take a `unit` argument (`--unit meaningful` or `--unit arbitrary`) and return, for every cut, both flags and a decision (`cut`, `flagged`, `undecided`); without a unit, a cut on which the two drops disagree is `undecided`. A no-cut drop above 0.05 is reported as a warning (`baseline_warning`). The flags themselves are unchanged from 4.8.0. Scripts keep their release prefixes, which earlier releases and archives cite; `scripts/INDEX.md` lists them by function.

## What is in version 4.8.1

Completes the analyses of 4.8.0: the measured cost of the refined tilt grid of Table 3 on the Junyi co-practice graph (`scripts/v11_refined_tilt_cost.py`, `results/v11/refined_tilt_cost.json`; 261.8 s under the global and 256.9 s under the row null, reproducing the detectable correlations of Table 3), and the grade-preserving baselines with the evaluation pairs kept out of the sampled non-edges on the Junyi co-practice graph as well (`results/v11/baselines_excluded.json`). No earlier result file changes.

## What is in version 4.8.0

Analyses that answer the sixth review of the Knowledge-Based Systems version of manuscript (A); new results in `results/v11/`, and `docs/paper_map_r6.md` maps each new item to its script and result file. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Cut audit of Algorithm 2 (step 3) on the thirteen real graphs over 20 seeds, both partitions of a drop re-drawn per seed, for the Q_F drop under x, x^3, x^(1/3) and 1-(1-x)^3 and for the rank-coded drop; ARI with the Junyi topics per seed | `scripts/v11_cut_audit_rescale.py` | `results/v11/cut_audit_rescale_<graph>.json` (Tables S3e, S3f) |
| Flagging rule over multipliers 2, 3 and minimum excesses 0, 0.005, 0.01, 0.02; agreement of seeds 0-9 and 10-19; bootstrap interval of the Junyi topic loss | `scripts/v11_flag_rules.py` | `results/v11/flag_rules.json` |
| Grade-preserving baselines retrained with every evaluation pair kept out of the sampled non-edges (needs PyTorch and the Junyi annotation files) | `fuzzykg/embeddings.py` (option `exclude`), `scripts/v11_baselines_excluded.py` | `results/v11/baselines_excluded.json` (Section S13) |

`fuzzykg.audit.cut_audit` and `scripts/fkg_audit.py` now flag a cut only when its excess over the no-cut drop also exceeds `min_excess` (default 0.01, the rule of the manuscript) and use 20 seeds by default; release 4.7.0 applied the 0.01 only in the manuscript. Release 4.8.0 supersedes `results/v7/qf_drop_seed_sd.json` and `results/v8/qf_drop_seeds.json` (Pi_F fixed at the best of five, ten seeds) as the source of the cut decisions.

## What is in version 4.7.0

Analyses that answer the fifth review of the Knowledge-Based Systems version of manuscript (A); new results in `results/v10/`, and `docs/paper_map_r5.md` maps each new item to its script and result file. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Membership-free link-prediction heuristics (common neighbours, Adamic-Adar, resource allocation, Jaccard, preferential attachment) on the hidden MOOCCubeX expert edges, on the folds and non-edges of the grade-preserving baselines | `scripts/v10_link_heuristics.py` | `results/v10/link_heuristics.json` |
| Paired bootstrap of the GCN, the UKGE-style model and the product path score against preferential attachment on the same folds (needs PyTorch) | `scripts/v10_gcn_vs_heuristics.py` | `results/v10/gcn_vs_heuristics.json` |
| Partition half of the audit on CN15k: description, Leiden timing, cut audit with the flagging rule of Algorithm 2 (Algorithm 1 not run; its estimated cost is reported) | `scripts/v10_cn15k_partition.py` | `results/v10/cn15k_partition.json` |
| Fig. 4 with the decision guide renumbered D1 to D9 and the degree heuristic in step 5 | `scripts/figure4.py` | `docs/figure4.png`, `docs/figure4.pdf` |

The manuscript of this release reads every cut decision from the Q_F drop (Algorithm 2, step 3; `results/v7/qf_drop_seed_sd.json`), restores the ten-graph decision floor of the Junyi co-practice graph and of the twelve transition variants (`results/v8/floor_extra*.json`) and the weighted SBM with 100 refits and the nested model with the covariate (`results/v8/weighted_sbm_*.json`), all produced by release 4.5.0. The CN15k raw files are read from `data/raw/ukge/cn15k` (clone https://github.com/stasl0217/UKGE).

## What is in version 4.6.0

Analyses that answer a further review of the Knowledge-Based Systems version of manuscript (A) (built on the third-revision manuscript of 4.4.0); new results in `results/v9/`, and `docs/paper_map_r4.md` maps each new item to its script and result file. No earlier result file changes.

| Topic | Script | Output |
|---|---|---|
| Grade-preserving baselines: a UKGE-style uncertain-KG embedding (ComplEx score, logistic confidence) and a GCN encoder with a bilinear decoder, trained to regress the memberships and tuned only on held-out edges, against the Junyi expert criteria and the hidden MOOCCubeX edges | `fuzzykg/embeddings.py`, `scripts/v9_grade_preserving_baselines.py` | `results/v9/grade_preserving_baselines.json` |
| Outside education: PPI5k (STRING protein interactions) and CN15k (ConceptNet) as released with UKGE; 10% of the pairs hidden and scored by every operator, the hop distance and both baselines; Algorithm 2 on PPI5k; percolation of the PPI5k cuts | `scripts/v9_cross_domain.py`, `scripts/v9_audit_cost.py ppi5k`, `scripts/v9_ppi_parallel_steps.py`, `scripts/v9_merge_ppi_audit.py`, `scripts/v9_ppi_percolation.py` | `results/v9/cross_domain_paths.json`, `results/v9/audit_cost_ppi5k.json`, `results/v9/ppi5k_percolation.json` |
| Measured time and peak memory of every step of Algorithm 2 on the largest graph (Junyi co-practice) | `scripts/v9_audit_cost.py junyi_copractice` | `results/v9/audit_cost_junyi_copractice.json` |
| Why max-min erases the graded evidence on the dense Junyi graph: reciprocity, percolation of the α-cuts, counterfactual graphs | `scripts/v9_maxmin_error.py` | `results/v9/maxmin_error.json` |
| Fig. 4, the audit from a log to a decision | `scripts/figure4.py` | `docs/figure4.png`, `docs/figure4.pdf` |

The baselines need PyTorch (`pip install -e .[baselines]`). The UKGE graphs are not redistributed: clone https://github.com/stasl0217/UKGE, copy `data/ppi5k` and `data/cn15k` to `data/raw/ukge/`, and run `python scripts/v9_cross_domain.py --write-graphs` before `scripts/v9_audit_cost.py ppi5k`. The PPI5k audit split steps 2d to 3 onto a second core (`scripts/v9_ppi_parallel_steps.py`, merged by `scripts/v9_merge_ppi_audit.py`), so its timings are not reported.

## What is in version 4.5.0

Analyses behind the third-revision manuscript as submitted; `docs/paper_map_r3.md` maps every item of that version to its script and result file. New results are in `results/v8/`; no earlier result file is changed or removed. Release 4.5.0 supersedes the nested-SBM and floor-variant files of 4.4.0 (`results/v7/weighted_sbm_nested_fair.json`, `results/v7/variant_floor.json`), which the manuscript no longer cites.

**Correction (nested SBM), completed.** Release 4.4.0 found that the nested runs carry no information on the weight channel and set them aside. The cause: releases 4.2.0 to 4.4.0 passed the covariate to `minimize_nested_blockmodel_dl` inside `state_args` (`base_type`, `rec`, `rec_types`), which graph-tool accepts with a `UserWarning` and ignores. `scripts/v5_weighted_sbm.py` now passes it through `base_state=WeightedBlockState, base_state_args=...` and asserts that every weighted fit, flat or nested, carries the covariate. With the covariate in place the nested model agrees with the flat one (ASSISTments 2017 KC-level co-practice: 10 level-0 blocks, row-null z = 288; Supplementary Table S11b). The flat fits were never affected.

| Topic | Script | Output |
|---|---|---|
| Flat weighted SBM with 100 refits under each null, every fit the best of five initialisations, one-sided empirical p stored with every null, on the three KC-level co-practice graphs and the assessed MOOCCubeX graph; 50 row-null refits on the full MOOCCubeX graph | `SBM_NULL=100 SBM_NULL_INIT=5 SBM_OUT=results/v8/weighted_sbm_100.json SBM_PART_DIR=results/v8/sbm_partitions python scripts/v5_weighted_sbm.py mooccubex_cs assist2009_kc assist2015_kc assist2017_kc`; `SBM_KINDS=row SBM_NULL=50 ... mooccubex_cs_full` | `results/v8/weighted_sbm_100.json`, `results/v8/weighted_sbm_50_full.json` (Table S11a) |
| Nested weighted SBM with the covariate, row null, five graphs | `SBM_NESTED=1 SBM_KINDS=row SBM_NULL=20 SBM_NULL_INIT=5 ...` | `results/v8/weighted_sbm_nested.json` (Table S11b) |
| Clipping of the logit at 0.01 instead of 1e-4 (no released membership reaches 1e-4 from 0 or 1) | `SBM_CLIP=0.01 ... mooccubex_cs` | `results/v8/weighted_sbm_clip.json` |
| Blocks of the refitted weighted SBMs held fixed: inside versus between, directional gap, block-pair eta^2 | `WHERE_PART_DIR=results/v8/sbm_partitions WHERE_OUT=results/v8/sbm_where.json python scripts/v6_sbm_where.py <graphs>` | `results/v8/sbm_where.json` |
| Decision floor with ten planted graphs per fraction and a bootstrap interval: Junyi co-practice graph (Table 3, Table S3h) and all twelve variants of the Junyi transition construction (Table S6c) | `scripts/v8_floor_extra.py` | `results/v8/floor_extra.json`, `results/v8/floor_extra_variants.json` |
| Seed dispersion of the Q_F drop and the flagging rule of Algorithm 2 (step 3) applied to it, against the rule on the cost index | `scripts/v8_qf_drop_seeds.py` | `results/v8/qf_drop_seeds.json` (Table S5d) |

New options of `scripts/v5_weighted_sbm.py`: `SBM_PART_DIR`, `SBM_KINDS` (with `SBM_CLIP` of 4.4.0); every null in its output carries `p_empirical` and the refit `values`. The SBM runs of 4.5.0 used graph-tool 3.9 (conda-forge), whose fits can differ by a few blocks from those of 4.3.0, so every refitted graph is reported from one release. The Junyi log of the transition variants was read from the EduData mirror of the PSLC DataShop release (`data/README.md`).

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

## Versions 2.0.0 to 3.9.0 (knowledge-tracing code, moved)

These versions added the directory `kt/` with the code and run files of a separate knowledge-tracing study (manuscript
*When Does a Concept Graph Help a Knowledge Tracer? Null-Graph Controls by Entry Point for KC-Level First Attempts*). From
version 5.4.2 that code is no longer part of this repository: it lives in its own repository,
https://github.com/Bangth2026/kt-null-graph-controls (version 4.0.0, https://doi.org/10.5281/zenodo.24492767). The fuzzy-graph
package, scripts and results of manuscript (A) did not change in these versions, except as listed in their own sections.

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
