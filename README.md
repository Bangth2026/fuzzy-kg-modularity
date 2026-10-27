# Modularity, interval type-2 states and weakest-link composition on fuzzy knowledge graphs

Reference implementation and reproduction scripts for

> Hai Bang Truong. *What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs built from learner logs.* Under revision for **Information Sciences** (earlier title: *Modularity, interval type-2 states and weakest-link composition on fuzzy knowledge graphs*). Saigon International University, Ho Chi Minh City, Vietnam. ORCID 0000-0001-5094-3128.

[![tests](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The paper answers three questions about a knowledge graph whose edges carry fuzzy memberships and whose nodes carry graded, uncertain states:

1. **Modularity for fuzzy graphs.** A functional `Q_F` for directed fuzzy graphs (crisp partition, fuzzy edges) that keeps Leiden's well-connectedness guarantee (Prop. 1), is bounded, aggregation-invariant, monotone and finitely convergent (Prop. 2), and *dominates* the α-cut average of crisp modularities by Chebyshev's integral inequality (Prop. 3, Cor. 1). Remark 3 / Algorithm 1 derive the smallest useful resolution from the normalised cut.
2. **Interval type-2 node states.** The Monte-Carlo-dropout interval `[p̄ − ησ, p̄ + ησ]` induces an IT2 membership over *unknown / partial / mastered*; the footprint of uncertainty is in closed form, and the usual type-1 collapse leaves its own footprint on 7.3 % of the state space and degenerates to universal rejection beyond the closed-form half-width `h* = (1 − τ_ZPD)·max(a, b)/2`.
3. **Path composition.** Max–min composition is the upper envelope of all t-norms (Fact 1); the Yager and Schweizer–Sklar families only erode below the weakest link; the consequence is measured on real graphs.


## Revision (October 2026): what changed and why

The referee report for the Information Sciences submission identified three
errors in the earlier version and asked for ground-truth experiments. The
code now contains the corrections; the numbers they produce are in
`results/` and are the ones cited in the revised manuscript.

| Concern | Correction | Where |
|---|---|---|
| The Corollary 1 gap is **not** the cost of binarising (constant-membership counterexample) | Identity gap = γ(1/c − 1)Σ a_κ b_κ verified; cost measured on planted partitions with γ refitted per level set, NMI against the truth | `fuzzykg/synthetic.py::planted_partition_fuzzy_graph`, `scripts/benchmark_binarisation.py`, `results/binarisation.json` |
| 2c/W is the threshold of **balanced** cuts only; Algorithm 1 could not fail | General threshold θ(B) = (c/W)/(a₁b₂ + a₂b₁); degree-preserving null-model z-score and a "no structure detected" outcome | `fuzzykg/modularity.py::bipartition_threshold_general`, `fuzzykg/resolution.py::null_model_zscore`, `adaptive_resolution_with_null`, `scripts/null_model_check.py` |
| Best paths under Archimedean t-norms are **shortest paths** (additive generators), not an exponential enumeration | `generator()` + Dijkstra / hop-limited Bellman–Ford, exact to 1e-15 against the old enumeration; ordinal-invariance check of max-min under random increasing rescalings | `fuzzykg/tnorms.py::generator, generator_best_paths`, `scripts/exact_tnorm_paths.py`, `results/tnorm_paths.json` |
| The cautious type-1 collapse is the author's own construction; baselines missing | Gates (i) point estimate, (ii) interval-image type-1, (iii) probability of ZPD, (iv) expected membership, compared with the three-valued IT2 gate on the raw bounds | `fuzzykg/fuzzification.py` (new `gate_*`), `scripts/gate_baselines.py`, `results/gate_baselines.json` |
| Graph statistics / reference partitions | components, reciprocity, membership quartiles, NMI with components or problem sets | `scripts/graph_stats.py` |

Runs on the real graphs (require the public data, see `data/README.md`):

```bash
python scripts/benchmark_binarisation.py --edges data/graphs/*_edges.csv   # Table 8, gamma refitted
python scripts/null_model_check.py       data/graphs/*_edges.csv           # Table 7, last row
python scripts/exact_tnorm_paths.py      data/graphs/assist2009_item_edges.csv  # Table 11, no hop limit
python scripts/graph_stats.py            data/graphs/*_edges.csv
```

The earlier bounded enumeration (`enumerate_best_paths`) is kept only as a
cross-check; `generator_best_paths` is the reference implementation.

## Installation

```bash
git clone https://github.com/Bangth2026/fuzzy-kg-modularity.git
cd fuzzy-kg-modularity
pip install -e ".[dev]"          # numpy, scipy, pandas, python-igraph, leidenalg, pytest
pytest                           # 13 property tests, ~2 s
```

Python ≥ 3.10. Everything runs on one CPU core; no GPU, no API keys.

## What reproduces without any download

| Script | Paper | Runtime | Needs data |
|---|---|---|---|
| `scripts/verify_propositions.py` | Props. 1–3, Cor. 1, Remark 3, Fact 1, §6.2 identities | seconds | no |
| `scripts/reproduce_section85.py` | §8.5, **Table 10**, 7.3 %, footprint widths, `h*` | seconds | no |
| `scripts/reproduce_table9.py` | §8.4, **Table 9** (cut-point sensitivity, 63 125 fuzzifications) | seconds | no |
| `scripts/reproduce_table5.py [--full]` | §8.2, **Table 5** (synthetic scalability to 20 000 nodes) | minutes (`--full` ≈ 10 min) | no |
| `scripts/run_leiden_grid.py --synthetic 59` | Tables 2/4-style run, Algorithm 1 | seconds | no |
| `scripts/tnorm_sweep.py --synthetic 371` | Tables 7/8-style sweep | seconds | no |

## What needs the public datasets

Download instructions and expected paths are in [`data/README.md`](data/README.md). Nothing is redistributed here.

```bash
# MOOCCubeX-CS (two small files from the THU-KEG release)         -> Table 2, 6, 8, 11 input
python scripts/build_mooccubex_graph.py
python scripts/run_leiden_grid.py data/graphs/mooccubex_cs_edges.csv --gammas 0.5 1.0 --seeds 10 --algorithm1
python scripts/tnorm_sweep.py   data/graphs/mooccubex_cs_edges.csv --sources all --hops 5 --standard-only

# ASSISTments 2009 / 2015 / 2017, Section 7.1 end to end          -> Table 3, 4, 7
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
  fuzzification.py  Eq. (10)–(13): collapse, footprint, Nie–Tan, three-valued gate, h*
  tnorms.py         min / product / Łukasiewicz, Yager, Schweizer–Sklar, OWA + orness → weights, widest path, bounded enumeration
  construction.py   Eq. (14)–(15): precedence, mastery gain, taxonomy prior, OWA aggregation, pruning
  synthetic.py      §8.2 generator (density 1.54, 40 topics / 1000 nodes, acyclic, Beta(2, 1.2) memberships)
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
| Table 10, type-1 pass column | 27.4 / 22.9 / 16.9 / 0 / 0 / 0 / 0 | identical |
| Table 10, IT2 pass / fail / indeterminate columns | all rows | identical |
| Table 10, Nie–Tan agreement column | 100 / 95.5 / 89.6 / 71.1 / 61.2 / 67.7 / 68.2 | identical |
| Table 10 aggregates | 9.6 / 10.7 / 39.5 / 30.6 / 79.0 | identical |
| Table 9, all 12 single-cut rows | — | identical to three decimals |
| Table 9, joint 125 settings | 0.132 / 6.0 %, worst 0.262 / 13.9 % | identical (see note 3) |
| §6.2 Yager(1) = Łukasiewicz | 1.1 × 10⁻¹⁶ | 1.1 × 10⁻¹⁶ |
| Props. 1–3, Cor. 1, Remark 3, Fact 1 | zero counter-examples | zero counter-examples |

Timings (2.79×, 4.23×, Table 5 seconds) are implementation- and hardware-specific and are not expected to match; the orders of magnitude do.

### Notes on reproduction

1. **Table 10 gate.** The submitted manuscript gates on the raw footprint bounds of Eq. (13) (`zpd_gate_it2`), which is well-formed (`lower ≤ upper` everywhere) and is what the table reports. An earlier draft normalised the lower-bound and upper-bound vectors separately onto the simplex before gating; that variant is kept as `zpd_gate_it2_normalised` for provenance and `reproduce_section85.py` prints it second. Separate normalisation does not preserve `lower ≤ upper`, so it double-counts nine grid points at ησ = 0.20 — the reason the draft's row summed to 104.5 % — and it is not used in the paper. The Nie–Tan column is the same under both (raw bounds averaged, then renormalised).
2. **Table 9, joint perturbation.** Of the 125 settings, the five with α_par + 0.10 and α_mas − 0.10 have α_par = α_mas = 0.70 in exact arithmetic (a zero-width mastered band, retained as a step); the script prints the figures with and without them (0.132 / 6.0 % vs 0.131 / 6.1 %).
3. **ASSISTments reconstruction.** The 2009 release has undisclosed upstream preprocessing, so Table 3 counts are reproducible but may differ in the last digits from the published build. Precedence is implemented as the directional share `N(a→b) / (N(a→b) + N(b→a))` within the horizon; "prior partial mastery of c_a" is a running accuracy on c_a of at least 0.5 before the attempt on c_b. Both are documented in `fuzzykg/construction.py`.
4. The Schweizer–Sklar limit p → 0 is implemented as the product (its analytic limit); p → −∞ and the Yager p → ∞ limit are approached numerically and checked at finite p with the expected `2^(1/p) − 1` deviation.

## Citation

```bibtex
@article{truong2026fuzzymodularity,
  author  = {Truong, Hai Bang},
  title   = {Modularity, interval type-2 states and weakest-link composition on fuzzy knowledge graphs},
  journal = {Fuzzy Sets and Systems},
  year    = {2026},
  note    = {submitted},
}
```

See also `CITATION.cff`.

## License

MIT for the code (see `LICENSE`). Datasets remain under their own licenses; see `data/README.md`.

## Acknowledgements

The ASSISTments team at Worcester Polytechnic Institute for the datasets, THU-KEG for MOOCCubeX, and the authors of `leidenalg` and `python-igraph`.

### Reproduction note (27 Oct 2026)

Rebuilt from the raw public files in a clean environment with the scripts above:
MOOCCubeX-CS matches Table 2 exactly (59 nodes, 91 edges, μ 0.400/0.934/0.999); the
five ASSISTments graphs match Table 3 in node and edge counts to within 0.3%
(2009 KC 110/5069, 2009 item 371/4947, 2015 KC 100/6963, 2017 KC 83/5494,
2017 item 867/33604) but their mean μ_E is 0.01–0.05 higher than Table 3, and
`spectral_bisection` returns c/W values different from Table 5. The original
Tables 2–5 and 11 came from an earlier script; `results/null_model.json`,
`results/binarisation_real.json`, `results/tnorm_paths_assist2009_item.json`
and `results/graph_stats.json` come from this repository. The loader fixes
needed to reach the counts above (JSON-lines `cs.json`, CS-discipline filter by
concept name, one log entry per (learner, order), missing-skill rows dropped at
item level) are in `data/loaders.py`.
