# Release 3.0.0: the learning-curve runs (revision R2)

Code, run files and analysis behind Sections 6.1 to 6.5 and 6.8 of the Neurocomputing manuscript
*When does a concept graph help a neural knowledge tracer?* (Tables 2, 3 and 6, Figs. 2 and 3,
the mixed model of Section 6.2, the gates of Section 6.4 and the prerequisite-consistency loss).

| Path | Content |
|---|---|
| `src/` | `train.py` (FKG-KT with empty-graph control, support graph, CAG and free-sign gates, pooled-cohort graph, prerequisite-consistency loss; DKT, SAKT, AKT, GKT, simpleKT, RKT), `models.py`, `graphs.py`, `prep.py`, `genjobs.py`, `runner.py`, analysis `r2_analysis.py`, `r2_stats.py`, `r2_perkc.py`, `r2_extra.py`, `r2_figures.py` |
| `runs/` | 1,069 run files (JSON); the 957 learning-curve runs are those with `src = R2` in `results/r2_runs.csv` |
| `results/` | `r2_runs.csv` (one row per run), `r2_results.json` (all comparisons with their Holm families), `r2_perkc.json`, `r2_extra.json`, `r2_table1.json` |
| `graphs/` | modularity-audit files of the candidate graphs (Supplementary S7) |
| `pyktp/` | runs under the pyKT protocol (Section 6.8) |

Scripts assume the working directory `/home/claude/kt` of the original run (`K` at the top of each
script); set it to your checkout and place the preprocessed corpora of `kt/src/prep.py` in `data/`.

Holm families in `r2_results.json`: `graph_vs_empty` holds two families, the 8 support-versus-control
tests (Table 3) and the 40 comparisons of the two gates and the pooled-cohort graph with the control
and with the support graph; `model_comparisons` holds the 24 probe-versus-baseline tests (Section 6.1);
`ablation` the 5 ablations of Fig. 2; `pc_loss` the 6 tests of the second channel.

Prerequisite-consistency loss (`train.py`, `--pc`, `--pclam`): for every prediction in the batch an
edge (a, b) is drawn uniformly from the graph; p(a) and p(b) come from two extra forward passes with
the target KC replaced by a and by b; the loss is lambda * mean(w_ab * relu(p(b) - p(a))).
Gate parameters (`gate` in the run file) were logged only for runs trained after logging was added:
21 of the 140 CAG runs.
