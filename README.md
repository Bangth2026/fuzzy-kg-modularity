# What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs

Reference implementation and reproduction scripts for

> Hai Bang Truong. *What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs built from learner logs.* Submitted to **Information Sciences**. Saigon International University, Ho Chi Minh City, Vietnam. ORCID 0000-0001-5094-3128.

[![tests](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/Bangth2026/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The paper asks, for three shortcuts that discard the grades on a fuzzy knowledge graph built from learner logs, whether the loss is bounded, computable from the graph, and large enough to matter:

1. **Binarising before partitioning.** The fuzzy-graph modularity `Q_F` (crisp partition, fuzzy edges) exceeds the α-average of its level-set surrogates under global normalisation by γ Σ Cov(f_κ, g_κ) (Proposition 2, Corollary 1); that gap is *not* the cost of binarising (Remark 3). The cost is measured on planted partitions with ground truth and on six real graphs, with the resolution refitted per level set: it is governed by the fuzzy mass the cut discards, not by the membership spread. Proposition 1 gives the resolution threshold of a bipartition; Algorithm 1 seeds the resolution from it and tests the result against a degree-preserving null model.
2. **Collapsing an uncertain state.** The interval `[p̄ − ησ, p̄ + ησ]` induces an interval type-2 membership over *unknown / partial / mastered* with a closed-form footprint; the cautious type-1 collapse rejects every candidate beyond `h* = (1 − τ_ZPD)·max(a, b)/2`, a defect of that construction that simpler gates do not share.
3. **Replacing max-min by an Archimedean t-norm.** Best paths under every continuous Archimedean t-norm are shortest-path problems through the additive generator (Section 7.3), so no family is computationally privileged; what distinguishes max-min is ordinal invariance (Proposition 3), verified on synthetic graphs and measured on real ones.

## What is in version 1.1

| Topic | What the code does | Where |
|---|---|---|
| The Corollary 1 gap vs the cost of binarising | Checks the constant-membership identity gap = γ(1/c − 1)Σ a_κ b_κ (Remark 3); measures the cost of binarising on directed planted partitions with controllable membership mean / spread / community shift, γ refitted per level set, NMI against the planted communities (Table 6); the same protocol on the real graphs (Table 8) | `fuzzykg/synthetic.py::planted_partition_fuzzy_graph`, `scripts/benchmark_binarisation.py`, `results/binarisation.json`, `results/binarisation_real.json` |
| Resolution threshold and Algorithm 1 | General bipartition threshold θ(B) = (c/W)/(a₁b₂ + a₂b₁) (Proposition 1); degree-preserving null model at a reference resolution with an effect-size floor and a "no structure detected" outcome (Table 7) | `fuzzykg/modularity.py::bipartition_threshold_general`, `fuzzykg/resolution.py::null_model_zscore`, `adaptive_resolution_with_null`, `scripts/null_model_check.py`, `results/null_model.json` |
| Best paths under Archimedean t-norms | Additive generators; exact best paths by Dijkstra / hop-limited Bellman–Ford (Section 7.3), checked to 1e-15 against bounded enumeration; ordinal-invariance check of max-min (Proposition 3); per-source Kendall τ-b with bootstrap CIs (Table 11b) | `fuzzykg/tnorms.py::generator, generator_best_paths`, `scripts/exact_tnorm_paths.py`, `results/tnorm_paths*.json` |
| Gates on the uncertain state | Point-estimate, interval-image type-1, probability-of-ZPD and expected-membership gates next to the cautious collapse and the three-valued IT2 gate (Table 13) | `fuzzykg/fuzzification.py` (`gate_*`), `scripts/gate_baselines.py`, `results/gate_baselines.json` |
| Graph statistics | Components, reciprocity, membership quartiles, NMI of the partition with components or problem sets | `scripts/graph_stats.py`, `results/graph_stats.json` |

Runs on the real graphs (require the public data, see `data/README.md`):

```bash
python scripts/benchmark_binarisation.py --edges data/graphs/*_edges.csv   # Table 8
python scripts/null_model_check.py       data/graphs/*_edges.csv           # Table 7
python scripts/exact_tnorm_paths.py      data/graphs/assist2009_item_edges.csv  # Table 11b
python scripts/graph_stats.py            data/graphs/*_edges.csv
```

The bounded enumeration (`enumerate_best_paths`) is kept only as a
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
  journal = {Information Sciences},
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
`spectral_bisection` returns c/W values different from Table 5. Tables 2–5 and 11 of the paper were produced with an earlier script; `results/null_model.json`,
`results/binarisation_real.json`, `results/tnorm_paths_assist2009_item.json`
and `results/graph_stats.json` come from this repository. The loader fixes
needed to reach the counts above (JSON-lines `cs.json`, CS-discipline filter by
concept name, one log entry per (learner, order), missing-skill rows dropped at
item level) are in `data/loaders.py`.
