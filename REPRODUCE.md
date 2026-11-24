# Reproducing the manuscript (release 5.4.0)

## Environment

Python 3.13.18 on Linux; exact package versions in `requirements-lock.txt` (NumPy 2.5.3, SciPy 1.18.1,
pandas 3.0.5, scikit-learn 1.9.1, python-igraph 1.0.0, leidenalg 0.12.0, numba 0.68.1). Install with

    pip install -r requirements-lock.txt && pip install -e .
    python -m pytest -q tests        # 55 tests

## Data

All inputs are public; `data/README.md` gives the source of every file. Place the raw logs under
`data/raw/` (ASSISTments 2009, 2012, 2015, 2017; KDD Cup 2010 Algebra I 2005-06 and Bridge to Algebra
2006-07; Eedi, NeurIPS 2020 Education Challenge; EdNet-KT1; Junyi Academy, PSLC DataShop 1198;
MOOCCubeX; UKGE PPI5k and CN15k). The derived graphs used by the paper are already in `data/graphs/`
(nodes and edges with memberships), so every analysis below that starts from a graph runs without the
raw logs; the Junyi crowd annotations come with the Junyi release.

## Seeds

Every random draw uses a seed fixed in the script that makes it (Leiden seeds 0-19; seed triples
0-2, ..., 15-17; bootstrap and permutation seeds are named in each script's docstring).

## Main text: table or figure -> script -> result file

Run times are wall-clock on one or two cores of a 2-core cloud machine; "-" where not recorded.

| Manuscript item | Script | Result file | Time |
|---|---|---|---|
| Table 1 (graphs) | `scripts/build_*`, `scripts/v15_partitions.py` | `data/graphs/*`, `results/v15/` | - |
| Table 2, Section 7 (semi-synthetic worlds) | `scripts/v16_semisynthetic.py`, `scripts/v16_summarise.py` | `results/v16/summary.json` | - |
| Section 7.3 (injected worlds, divergence check) | `scripts/v16_injected.py`, `scripts/v17_divergence_null.py` | `results/v16/`, `results/v17/` | - |
| Table 3 (decomposition) | `scripts/v15_value_paths.py`, `scripts/v16_junyi_pruning.py`, `scripts/v16_real.py` | `results/v15/value_paths.json`, `results/v16/junyi_pruning_value.json`, `results/v16/real.json` | - |
| Table 4 (reach and order) | `scripts/v18_reach.py` | `results/v18/reach.json` | 10 min |
| Section 8.1 (three-level reach score, threshold chosen on held-out exercises, Table S9c scheme) | `scripts/v19_misc.py` | `results/v19/misc.json` | 5 min |
| Section 8.1 (cross-fitted best crisp vs best graded, Table S4g) | `scripts/v18_crossfit.py` | `results/v18/crossfit.json` | 2 min |
| Section 8.1 (partition value over B_2 and seed triples, Table S4i) | `scripts/v18_partition.py box`, `scripts/v19_interior.py` | `results/v18/partition.json`, `results/v19/interior.json` | 45 min + 35 min, 2 cores |
| Section 8.1 (Junyi without suggested attempts, Table S4h) | `scripts/v18_partition.py unsuggested` | `results/v18/partition.json` | 10 min |
| Table 5, Section 8.2 (robust decisions, external check) | `scripts/v16_real.py`, `scripts/v17_summarise.py` | `results/v17/summary.json` | - |
| Table 6, Section 8.3 (paths) | `scripts/v15_value_paths.py`, `scripts/v15_crossdomain.py` | `results/v15/` | - |
| Table 7, Section 8.4 (guide in use) | `scripts/v19_kbs.py` (next-concept inputs: `scripts/heldout_transitions.py`) | `results/v19/kbs.json`, `results/heldout_*.json` | 4 min |
| Table 8, Section 8.5 (primary results) | `scripts/v19_kbs.py`, `scripts/v19_robust.py perm`, `scripts/v19_interior.py` | `results/v19/` | 5 min |
| Section 9.3 (noise, scale; Tables S13c, S13d), Table S4j | `scripts/v19_robust.py noise scaling features` | `results/v19/robust.json` | 15 min |
| Figures 1 to 4 | `scripts/v16_figures.py`, `scripts/v15_figures.py` | `docs/v52_figure*` | - |

Every new number of release 5.4.0 is collected in `results/v19/` (`misc.json`, `interior.json`,
`kbs.json`, `robust.json`); those of 5.3.0 in `results/v18/summary.json`.
