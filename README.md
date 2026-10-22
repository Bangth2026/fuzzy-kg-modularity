# Modularity, interval type-2 states and weakest-link composition on fuzzy knowledge graphs

Reference implementation and reproduction scripts for

> Hai Bang Truong. *Modularity, interval type-2 states and weakest-link composition on fuzzy knowledge graphs.* Submitted to **Fuzzy Sets and Systems**. Saigon International University, Ho Chi Minh City, Vietnam. ORCID 0000-0001-5094-3128.

[![tests](https://github.com/<user>/fuzzy-kg-modularity/actions/workflows/tests.yml/badge.svg)](https://github.com/<user>/fuzzy-kg-modularity/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The paper answers three questions about a knowledge graph whose edges carry fuzzy memberships and whose nodes carry graded, uncertain states:

1. **Modularity for fuzzy graphs.** A functional `Q_F` for directed fuzzy graphs (crisp partition, fuzzy edges) that keeps Leiden's well-connectedness guarantee (Prop. 1), is bounded, aggregation-invariant, monotone and finitely convergent (Prop. 2), and *dominates* the α-cut average of crisp modularities by Chebyshev's integral inequality (Prop. 3, Cor. 1). Remark 3 / Algorithm 1 derive the smallest useful resolution from the normalised cut.
2. **Interval type-2 node states.** The Monte-Carlo-dropout interval `[p̄ − ησ, p̄ + ησ]` induces an IT2 membership over *unknown / partial / mastered*; the footprint of uncertainty is in closed form, and the usual type-1 collapse leaves its own footprint on 7.3 % of the state space and degenerates to universal rejection beyond the closed-form half-width `h* = (1 − τ_ZPD)·max(a, b)/2`.
3. **Path composition.** Max–min composition is the upper envelope of all t-norms (Fact 1); the Yager and Schweizer–Sklar families only erode below the weakest link; the consequence is measured on real graphs.

## Installation

```bash
git clone https://github.com/<user>/fuzzy-kg-modularity.git
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
| Table 10, IT2 pass / fail / indet. (as-published variant) | all rows | identical except row ησ = 0.20 (see below) |
| Table 10, Nie–Tan agreement column | 100 / 95.5 / 89.6 / 71.1 / 61.2 / 67.7 / 68.2 | identical |
| Table 10 aggregates | 9.6 / 17.3 / 17.6 / 30.6 / 79.0 | identical |
| Table 9, all 12 single-cut rows | — | identical to three decimals |
| Table 9, joint 125 settings | 0.132 / 6.0 %, worst 0.262 / 13.9 % | identical (see note 3) |
| §6.2 Yager(1) = Łukasiewicz | 1.1 × 10⁻¹⁶ | 1.1 × 10⁻¹⁶ |
| Props. 1–3, Cor. 1, Remark 3, Fact 1 | zero counter-examples | zero counter-examples |

Timings (2.79×, 4.23×, Table 5 seconds) are implementation- and hardware-specific and are not expected to match; the orders of magnitude do.

### Known discrepancies and notes for the author

1. **Table 10, row ησ = 0.20 sums to 104.5 %** (4.5 + 62.2 + 37.8). The published IT2 columns are reproduced exactly only when the *lower-bound vector and the upper-bound vector are each normalised onto the simplex before gating* (`zpd_gate_it2_normalised`). Separate normalisation does not preserve `lower ≤ upper`: at nine grid points (p̄\* ∈ {0.555, …, 0.595}, ησ = 0.20) the normalised lower bound of *partial* is 1 while its normalised upper bound is below τ, so those states were counted both as *pass* and as *fail*. Section 5.2 of the manuscript describes the gate on the raw bounds of Eq. (13), which is well-formed and gives the second variant printed by `reproduce_section85.py` (pass / fail / indeterminate = 22.9 / 68.7 / 8.5 at 0.02, …, 0 / 15.4 / 84.6 at 0.30; aggregate IT2 pass 10.7 %, indeterminate 39.5 %). The manuscript should either (a) state the normalisation and correct the 0.20 row and the derived sentences (“22.9 %, 14.4 % and 4.5 % passing”, “37.8 % of the state space”, “17.3 % pass and 17.6 % indeterminate”), or (b) adopt the raw-bound gate and update those numbers. The Nie–Tan column is unaffected (it uses the raw bounds, averaged then renormalised).
2. **“Footprint is well formed: no case with μ̲ > μ̄”** is true of the raw bounds (verified here) but not of the separately normalised bounds used for the published gate.
3. **Table 9, “all 125 settings remain well-ordered”.** In exact arithmetic the five settings with α_par + 0.10 and α_mas − 0.10 have α_par = α_mas = 0.70 (a zero-width mastered band); they count as ordered only because `0.60 + 0.10 > 0.80 − 0.10` by one ulp in floating point. Excluding them gives 0.131 / 6.1 %, worst 0.262 / 14.1 % (120 settings). Either number is fine; the sentence should say 120, or say that the degenerate band is handled as a step.
4. **ASSISTments reconstruction.** The 2009 release has undisclosed upstream preprocessing, so Table 3 counts are reproducible but may differ in the last digits from the published build. Precedence is implemented as the directional share `N(a→b) / (N(a→b) + N(b→a))` within the horizon; "prior partial mastery of c_a" is a running accuracy on c_a of at least 0.5 before the attempt on c_b. Both are documented in `fuzzykg/construction.py`.
5. The Schweizer–Sklar limit p → 0 is implemented as the product (its analytic limit); p → −∞ and the Yager p → ∞ limit are approached numerically and checked at finite p with the expected `2^(1/p) − 1` deviation.

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
