# Map of the revised manuscript (release 5.1.0) to scripts and result files

| Manuscript item | Script | Result file / key |
|---|---|---|
| Abstract, Section 1 (Junyi values before and after pruning) | `scripts/v16_junyi_pruning.py` | `results/v16/junyi_pruning_value.json` |
| Corollary 1; Supplementary S2 (refined family) | `scripts/v16_real.py` | `results/v16/real.json` (`kappa_star_refined`) |
| Section 4.5, Eq. (7) (divergence check) | `fuzzykg/simplex.py`, `scripts/v16_divergence.py` | `results/v16/divergence.json` |
| Table 2, Section 7.1 (balanced accuracy, constant rule, test worlds, cross-validated delta, loss against Pi_F) | `scripts/v16_summarise.py` | `summary.json` → `semisynthetic_v15` |
| Section 7.3; Supplementary Tables S3g, S3h | `scripts/v16_semisynthetic.py`, `scripts/v16_injected.py`, `scripts/v16_divergence.py` | `summary.json` → `semisynthetic_delta`, `injected` |
| Section 8.1, Table 3 (pruning rows, seed-triple ranges), Fig. 4; Supplementary Tables S4b, S4c, S4e | `scripts/v16_junyi_pruning.py`, `scripts/v16_units.py`, `scripts/v16_real.py`, `scripts/v16_figures.py` | `junyi_pruning_value.json`, `units.json`, `real.json` |
| Section 8.2 (external check against both losses, checks, seed resamples of kappa*, learner resampling); Supplementary Tables S5b to S5d, S7b | `scripts/v16_real.py`, `scripts/v16_learner_resampling.py` | `summary.json` → `real`, `learner_resampling` |
| Table 4 (kappa* reported as "> 20") | `scripts/v15_summarise.py` | `results/v15/summary.json` |
| Table 5 (lambda-hat shown only where identified) | `scripts/v15_value_paths.py` | `results/v15/value_paths.json` (`argmax_lam_ci`) |
| Figures 1 to 4 | `scripts/v16_figures.py` (Fig. 3 from `scripts/v15_figures.py`) | `docs/v51_figure*` |
