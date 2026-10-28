# Binarisation and t-norm composition on fuzzy knowledge graphs

Reference implementation and reproduction scripts for

> Hai Bang Truong. *Binarisation and t-norm composition on fuzzy knowledge graphs: a multislice reference, a calibrated null test and expert-labelled criteria.* Submitted to **Information Sciences**. Saigon International University, Ho Chi Minh City, Vietnam. ORCID 0000-0001-5094-3128.

[![tests](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23689762.svg)](https://doi.org/10.5281/zenodo.23689762)

A fuzzy knowledge graph built from learner logs carries graded prerequisite edges. Pipelines that consume it discard the grades at two points: they threshold the graph before running a crisp community detector, and they replace max-min composition along paths by a non-idempotent t-norm. This repository measures what each shortcut costs.

1. **Binarising before partitioning.** The multislice modularity of the α-cut family with one shared partition (`Qbar`) is the reference; the cost index `cost(α₀) = Qbar(Π*) − Qbar(Π_α₀)` prices a single cut. The index is checked against a recovery loss defined from the ground truth alone, on planted partitions and an LFR-style benchmark, next to the free proxies (retained mass, mean out-degree, singleton share, crisp modularity).
2. **What the grades carry.** A calibrated null test (`fuzzykg/nulltest.py`): a degree-preserving null with a decision floor read off planted graphs of the same size, density and number of communities; two weights-only nulls (global, and row-wise keeping every out-strength); and a power calibration that tilts the observed memberships toward internal edges on the observed topology, so that a null result has a detectable effect size.
3. **Composition.** Best paths under every continuous Archimedean t-norm are shortest paths through the additive generator; all-pairs best-path strengths for ten operators, with optional hop limit and path floor (`fuzzykg/paths_fast.py`). The operators are judged against expert prerequisite labels (Junyi Academy crowd scores and knowledge map; MOOCCubeX held-out expert edges) and against the learner's next concept in held-out logs, with direct-edge, bigram and popularity baselines and paired bootstrap intervals over learners.

Every analysis is run on graphs from two constructions of the same logs: a **co-practice** construction (precedence share and mastery gain, OWA-aggregated; dense, nearly reciprocal) and an **order-preserving transition** construction (directional dominance and successor preference of consecutive runs, Gödel conjunction; sparse, no reciprocal edges).

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
  title   = {Binarisation and t-norm composition on fuzzy knowledge graphs: a multislice reference, a calibrated null test and expert-labelled criteria},
  journal = {Information Sciences},
  year    = {2026},
  note    = {submitted},
}
```

Archived on Zenodo: https://doi.org/10.5281/zenodo.23689762 (all versions; resolves to the latest). See also `CITATION.cff`.

## License

MIT for the code (see `LICENSE`). Datasets remain under their own licenses (Junyi Academy: non-commercial use, cite Chang, Hsu and Chen, EDM 2015); see `data/README.md`.

## Acknowledgements

The ASSISTments team at Worcester Polytechnic Institute, THU-KEG for MOOCCubeX, Junyi Academy and the authors of its public release, and the authors of `leidenalg` and `python-igraph`.
