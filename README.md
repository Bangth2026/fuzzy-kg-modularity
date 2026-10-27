# Level-set modularity for fuzzy knowledge graphs

Reference implementation and reproduction scripts for

> Hai Bang Truong. *Level-set modularity for fuzzy knowledge graphs: pricing binarisation and non-idempotent t-norms against planted partitions, null models and held-out transitions.* Submitted to **Information Sciences**. Saigon International University, Ho Chi Minh City, Vietnam. ORCID 0000-0001-5094-3128.

[![tests](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23689763.svg)](https://doi.org/10.5281/zenodo.23689763)

A fuzzy knowledge graph built from learner logs carries graded prerequisite edges. The pipelines that consume it discard the grades at two points, and this repository prices both shortcuts:

1. **Binarising before partitioning.** The *level-set modularity* `Qbar` (Section 5.5 of the paper) is the multislice modularity whose slices are the α-cuts of the fuzzy graph, with one partition shared by all slices; it is optimised with the multiplex Leiden procedure, relates exactly to the fuzzy-graph modularity `Q_F` (Lemma 3), and prices a single-cut partition through the cost index `cost(α₀) = Qbar(Π*) − Qbar(Π_α₀)`. The index is tested against a planted truth with the community information in the memberships swept from none to full, on an LFR-style directed benchmark, and on seven real graphs; four null models (degree-, order- and component-preserving, and weights-only) decide what the grades of the real graphs carry.
2. **Replacing max-min by an Archimedean t-norm.** Best paths under every continuous Archimedean t-norm are shortest paths through the additive generator (Section 6.3); max-min is the one operator whose ordering survives every increasing rescaling (Proposition 3); and held-out next-concept prediction on the log-built graphs supplies the external criterion, with the Yager and Schweizer–Sklar parameters chosen on a validation split.

The interval-valued learner state and its cautious type-1 collapse are treated in Supplementary Section S1 of the paper and kept here as a representation (`fuzzykg.fuzzification`).

## What is in version 1.3

Version 1.3 adds, on top of version 1.2: a conditional (bigram) baseline, reachability and one-hop diagnostics in the held-out experiment; a calibration of the effect-size floor of the null-model test on dense planted graphs and a 50-seed check of the weights-only null; a disparity-filter backbone (`fuzzykg/backbone.py`) as a comparison cut; sensitivity of the level-set objective and the cost index to the number of levels and to the optimiser seed; and five graph seeds (previously three) on the fixed-information planted benchmark, with the run splittable over processes (`--seed-list`, `--merge`).

| Topic | What the code does | Where |
|---|---|---|
| Level-set modularity | `Qbar` over a quantile grid of α-cuts (mass or equal layer weights), multiplex Leiden optimiser, per-level profile, cost index, persistence tracks along the α-filtration | `fuzzykg/multislice.py`, `scripts/benchmark_levelset.py`, `scripts/levelset_real.py`, `results/levelset_{shift,lfr,real}.json` |
| Membership-information sweep | Planted partitions with the community shift of the memberships swept 0 → 1, ten seeds; Fuzzy Leiden, level-set (both weightings), Algorithm 1 and single cuts compared by ARI; cost index vs recovery loss (Table 8) | `scripts/benchmark_levelset.py --stage shift` |
| LFR-style benchmark | Directed generator with power-law community sizes and out-degrees, mixing 0.1–0.6, sizes 500 / 2000 / 10000 (Table S1, supplementary) | `fuzzykg/synthetic.py::lfr_style_fuzzy_graph`, `scripts/benchmark_levelset.py --stage lfr` |
| Four null models | Degree-preserving, curriculum-order-preserving, component-preserving and weights-only nulls at the reference resolution, 200 draws on the real graphs (Table 6) | `scripts/null_models_extended.py`, `results/null_models_extended.json` |
| Held-out next-concept prediction | Learner split 70/10/20, graph rebuilt on training learners, MRR / hit@k of the true next concept under each composition operator, family parameters selected on validation; baselines: direct edge, popularity, training bigram, random; diagnostics: unreachable share, dropped transitions, one-hop share and τ-b against the direct-edge ranking per operator (Table 13) | `scripts/heldout_transitions.py`, `results/heldout_*.json` |
| Effect-size floor | Outcome of the null test at floors 0.02 / 0.05 / 0.10 and the relative excess; calibration of the floor on dense planted graphs (110 nodes, 50 edges per node, p_in swept); observed Q_F with 50 optimiser seeds against the weights-only null (Table S2 and Section 8.1) | `scripts/null_floor_calibration.py`, `results/null_floor_*.json` |
| Sensitivity of the level-set objective | Cost index on grids of 4 / 8 / 16 quantile levels, Q̄(Π*) dispersion over ten optimiser seeds, disparity-filter backbone at significance 0.05 / 0.20 on the planted sweep and the real graphs; stratified correlations of the cost index with the recovery loss; optimiser-seed dispersion on the LFR-style benchmark (Table S3) | `fuzzykg/backbone.py`, `scripts/levelset_sensitivity.py`, `results/levelset_sensitivity_*.json` |
| Full curated graph | MOOCCubeX-CS with all 725 expert pairs (287 concepts) next to the assessed 59-concept graph | `scripts/build_mooccubex_graph.py --full` |
| The Proposition 2 gap vs the cost of binarising | Constant-membership identity (Remark 3); the shortcut on planted partitions with γ refitted per level set, five graph seeds (Table 7) and on the real graphs (Table 9) | `scripts/benchmark_binarisation.py`, `results/binarisation*.json` |
| Resolution threshold and Algorithm 1 | General bipartition threshold θ(B) (Proposition 1); seeded resolution with a null-model test and a "no structure detected" outcome (Table 5) | `fuzzykg/modularity.py`, `fuzzykg/resolution.py`, `scripts/null_model_check.py` |
| Best paths under Archimedean t-norms | Additive generators; exact best paths by Dijkstra / hop-limited Bellman–Ford, checked against bounded enumeration; ordinal-invariance check (Proposition 3); per-source Kendall τ-b with bootstrap CIs (Table 12) | `fuzzykg/tnorms.py`, `scripts/exact_tnorm_paths.py`, `scripts/composition_real.py` |
| Gates on the uncertain state | Cautious collapse, footprint, three-valued IT2 gate and four comparison gates (Supplementary Section S1, Table S5) | `fuzzykg/fuzzification.py`, `scripts/gate_baselines.py` |

Runs on the real graphs (require the public data, see `data/README.md`):

```bash
python scripts/levelset_real.py                                            # Table 9 (level-set columns)
python scripts/null_models_extended.py --draws 200 data/graphs/*_edges.csv # Table 6
python scripts/heldout_transitions.py --release 2009 --level item          # Table 13 (one column)
python scripts/levelset_sensitivity.py --part real                         # Table S3 (real graphs)
python scripts/null_floor_calibration.py --part seeds                      # 50-seed observed Q_F vs weights-only null
python scripts/benchmark_binarisation.py --edges data/graphs/*_edges.csv   # Table 9
python scripts/null_model_check.py       data/graphs/*_edges.csv           # Table 5
python scripts/real_graph_tables.py                                        # Tables 2-4
python scripts/composition_real.py                                         # Table 12
python scripts/tnorm_sweep.py data/graphs/assist2009_item_edges.csv --sources 60 --hops 4   # Table 11
```

Synthetic runs (no download):

```bash
python scripts/benchmark_levelset.py --stage shift --seeds 10             # Table 8
python scripts/benchmark_levelset.py --stage lfr   --seeds 10             # Table S1
python scripts/benchmark_binarisation.py --graph-seeds 5                  # Table 7 (one process)
# or split over processes and merge:
#   python scripts/benchmark_binarisation.py --seed-list 0 1 2 --out results/binarisation_a.json
#   python scripts/benchmark_binarisation.py --seed-list 3 4   --out results/binarisation_b.json
#   python scripts/benchmark_binarisation.py --merge results/binarisation_a.json results/binarisation_b.json
python scripts/levelset_sensitivity.py --part shift                       # Table S3 (planted sweep)
python scripts/levelset_sensitivity.py --part lfr_seeds                   # optimiser-seed dispersion, LFR-style
python scripts/levelset_sensitivity.py --part correlations                # cost index vs recovery loss, stratified
python scripts/null_floor_calibration.py --part planted                   # Table S2
python scripts/null_floor_calibration.py --part floors                    # outcomes at three floors
```

## Installation

```bash
git clone https://github.com/Bangth2026/fuzzy-kg-modularity.git
cd fuzzy-kg-modularity
pip install -e ".[dev]"          # numpy, scipy, pandas, python-igraph, leidenalg, pytest
pytest                           # 28 property tests, ~2 s
```

Python ≥ 3.10. Everything runs on one CPU core; no GPU, no API keys.

## What reproduces without any download

| Script | Paper | Runtime | Needs data |
|---|---|---|---|
| `scripts/verify_propositions.py` | Props. 1–3, Cor. 1, Remark 3, Fact 1, §6.2 identities, Appendix B identities | seconds | no |
| `scripts/reproduce_section85.py` | Appendix B, **Table S5**, footprint widths, `h*` | seconds | no |
| `scripts/reproduce_table9.py` | Appendix B, **Table S4** (cut-point sensitivity, 63 125 fuzzifications) | seconds | no |
| `scripts/reproduce_table5.py [--full]` | §8.4, **Table 10** (synthetic scalability to 20 000 nodes) | minutes (`--full` ≈ 10 min) | no |
| `scripts/run_leiden_grid.py --synthetic 59` | Tables 2/4-style run, Algorithm 1 | seconds | no |
| `scripts/tnorm_sweep.py --synthetic 371` | Tables 7/8-style sweep | seconds | no |

## What needs the public datasets

Download instructions and expected paths are in [`data/README.md`](data/README.md). Nothing is redistributed here.

```bash
# MOOCCubeX-CS (two small files from the THU-KEG release)         -> Tables 2, 5, 6, 9, 12 input
python scripts/build_mooccubex_graph.py            # assessed graph (59 concepts)
python scripts/build_mooccubex_graph.py --full     # full expert graph (287 concepts)
python scripts/run_leiden_grid.py data/graphs/mooccubex_cs_edges.csv --gammas 0.5 1.0 --seeds 10 --algorithm1
python scripts/tnorm_sweep.py   data/graphs/mooccubex_cs_edges.csv --sources all --hops 5 --standard-only

# ASSISTments 2009 / 2015 / 2017, Section 7.1 end to end          -> Tables 3, 4, 5, 6, 9, 12, 13
python scripts/build_assistments_graph.py --release 2009 --level kc
python scripts/build_assistments_graph.py --release 2009 --level item
python scripts/build_assistments_graph.py --release 2015 --level kc
python scripts/build_assistments_graph.py --release 2017 --level kc
python scripts/build_assistments_graph.py --release 2017 --level item
python scripts/run_leiden_grid.py data/graphs/assist2009_item_edges.csv --gammas 0.5 1 2 5 --seeds 5 --algorithm1
python scripts/tnorm_sweep.py   data/graphs/assist2009_item_edges.csv --sources 60 --hops 4
# orness sweep of Section 8.4
for o in 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do python scripts/build_assistments_graph.py --release 2009 --level kc --orness $o; done
```

## Package layout

```
fuzzykg/
  graph.py          FuzzyGraph: directed, μ_E ∈ (0,1], α-cuts, strengths, reciprocity
  modularity.py     Q_F (Eq. 7–8), Q̃(α) (Eq. 9), exact α-average, Cor. 1 gap and bounds, Remark 3 threshold
  leiden.py         Fuzzy Leiden via leidenalg (RBConfiguration, directed, weighted) + Prop. 1 / Prop. 2(iii) checks
  resolution.py     Algorithm 1 (spectral bisection → 2c/W → seed-stability loop), ARI
  fuzzification.py  Eq. (S.1)–(S.4) of the Supplementary material: collapse, footprint, Nie–Tan, three-valued gate, h*
  tnorms.py         min / product / Łukasiewicz, Yager, Schweizer–Sklar, OWA + orness → weights, widest path, bounded enumeration
  construction.py   Eq. (12)–(13): precedence, mastery gain, taxonomy prior, OWA aggregation, pruning
  synthetic.py      scalability generator, planted partitions with membership shift, LFR-style directed benchmark
  multislice.py     level-set modularity Qbar, multiplex Leiden, cost index, persistence tracks
  backbone.py       disparity-filter backbone (Serrano et al. 2009) on a directed fuzzy graph
data/loaders.py     ASSISTments 2009/2015/2017 and MOOCCubeX-CS loaders with the stated preprocessing
scripts/            reproduction scripts listed above
tests/              pytest property tests
```

Minimal use:

```python
from fuzzykg import FuzzyGraph, fuzzy_leiden, correlation_gap, footprint, zpd_gate_it2, widest_path

G = FuzzyGraph.from_edges([(0, 1, 0.9), (1, 2, 0.4), (0, 2, 0.7)])
r = fuzzy_leiden(G, gamma=0.5, seed=0)          # r.membership, r.Q_F, r.connected_violations == 0
correlation_gap(G, r.membership, gamma=0.5)     # Q_F, α-average, gap = γ Σ Cov(f_k, g_k), bounds
footprint(p=0.55, halfwidth=0.1)                # {'unk': (lo, hi), 'par': (lo, hi), 'mas': (lo, hi)}
zpd_gate_it2(p=0.55, halfwidth=0.1, tau_zpd=0.4)  # 1 pass / 0 fail / -1 indeterminate
widest_path(G, source=0)                        # μ_E^∞(0, ·) by max–min Dijkstra
```

## Reproduction status

Values obtained with this code on the offline parts, against the manuscript:

| Quantity | Paper | This code |
|---|---|---|
| Collapse embedded in footprint (par) | 92.7 % (7.3 % leave) | 92.7 % |
| Mean footprint width unk / par / mas | 0.233 / 0.362 / 0.230 | 0.233 / 0.362 / 0.230 |
| `h*` at τ_ZPD = 0.4 | 0.075 | 0.075 |
| Table S5, type-1 pass column | 27.4 / 22.9 / 16.9 / 0 / 0 / 0 / 0 | identical |
| Table S5, IT2 pass / fail / indeterminate columns | all rows | identical |
| Table S5, Nie–Tan agreement column | 100 / 95.5 / 89.6 / 71.1 / 61.2 / 67.7 / 68.2 | identical |
| Table S5 aggregates | 9.6 / 10.7 / 39.5 / 30.6 / 79.0 | identical |
| Table S4, all 12 single-cut rows | — | identical to three decimals |
| Table S4, joint 125 settings | 0.132 / 6.0 %, worst 0.262 / 13.9 % | identical (see note 3) |
| §6.2 Yager(1) = Łukasiewicz | 1.1 × 10⁻¹⁶ | 1.1 × 10⁻¹⁶ |
| Props. 1–3, Cor. 1, Remark 3, Fact 1 | zero counter-examples | zero counter-examples |

Timings (Table 10 seconds) are implementation- and hardware-specific and are not expected to match; the orders of magnitude do.

### Notes on reproduction

1. **Table S5 gate.** The manuscript gates on the raw footprint bounds of Eq. (S.4) (`zpd_gate_it2`), which is well-formed (`lower ≤ upper` everywhere) and is what the table reports. An earlier draft normalised the lower-bound and upper-bound vectors separately onto the simplex before gating; that variant is kept as `zpd_gate_it2_normalised` for provenance and `reproduce_section85.py` prints it second. Separate normalisation does not preserve `lower ≤ upper`, so it double-counts nine grid points at ησ = 0.20 — the reason the draft's row summed to 104.5 % — and it is not used in the paper. The Nie–Tan column is the same under both (raw bounds averaged, then renormalised).
2. **Table S4, joint perturbation.** Of the 125 settings, the five with α_par + 0.10 and α_mas − 0.10 have α_par = α_mas = 0.70 in exact arithmetic (a zero-width mastered band, retained as a step); the script prints the figures with and without them (0.132 / 6.0 % vs 0.131 / 6.1 %).
3. **ASSISTments reconstruction.** The 2009 release has undisclosed upstream preprocessing, so Table 3 counts are reproducible but may differ in the last digits from the published build. Precedence is implemented as the directional share `N(a→b) / (N(a→b) + N(b→a))` within the horizon; "prior partial mastery of c_a" is a running accuracy on c_a of at least 0.5 before the attempt on c_b. Both are documented in `fuzzykg/construction.py`.
4. The Schweizer–Sklar limit p → 0 is implemented as the product (its analytic limit); p → −∞ and the Yager p → ∞ limit are approached numerically and checked at finite p with the expected `2^(1/p) − 1` deviation.

## Citation

```bibtex
@article{truong2026fuzzymodularity,
  author  = {Truong, Hai Bang},
  title   = {Level-set modularity for fuzzy knowledge graphs: pricing binarisation and non-idempotent t-norms against planted partitions, null models and held-out transitions},
  journal = {Information Sciences},
  year    = {2026},
  note    = {submitted},
}
```

Archived release v1.2.0: https://doi.org/10.5281/zenodo.23689763 (all versions: https://doi.org/10.5281/zenodo.23689762). See also `CITATION.cff`.

## License

MIT for the code (see `LICENSE`). Datasets remain under their own licenses; see `data/README.md`.

## Acknowledgements

The ASSISTments team at Worcester Polytechnic Institute for the datasets, THU-KEG for MOOCCubeX, and the authors of `leidenalg` and `python-igraph`.

### Reproduction note (27 Oct 2026)

All numbers in the paper come from this repository. Rebuilt from the raw public
files in a clean environment: MOOCCubeX-CS has 59 nodes and 91 edges (μ
0.400/0.934/0.999) in its assessed form and 287 nodes and 725 edges in its full
form; the ASSISTments graphs have 2009 KC 110/5069, 2009 item
371/4947, 2015 KC 100/6963, 2017 KC 83/5494, 2017 item 867/33604 nodes/edges.
The loader details that produce these counts (JSON-lines `cs.json` with the
CS-discipline filter by concept name, one log entry per (learner, order),
missing-skill rows dropped at item level) are in `data/loaders.py`.
