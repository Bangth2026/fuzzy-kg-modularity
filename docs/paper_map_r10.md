# Release 5.0.0: manuscript item -> script -> result

| Manuscript item | Script | Result file / key |
|---|---|---|
| Fig. 1 | scripts/v15_figures.py | docs/v5_figure1 |
| Proposition 1 check, Algorithm 1 cost | scripts/v15_checks.py | results/v15/checks.json |
| Section 7, Table 2, Fig. 2 | scripts/v15_semisynthetic.py, v15_summarise.py | summary.json: semisynthetic |
| Section 4.4 (minimum excess) | scripts/v15_summarise.py | summary.json: delta, semisynthetic.delta_curve_* |
| Section 8.1, Table 3 | scripts/v15_value_paths.py, v15_crossdomain.py | value_paths.json, crossdomain.json |
| Section 8.2, Table 4, external check | scripts/v15_simplex_real.py, v15_summarise.py | summary.json: real |
| Section 8.2, learner bootstrap | scripts/v15_learner_bootstrap.py | learner_bootstrap.json, summary.json: bootstrap |
| Section 8.3, Table 5, Fig. 3 | scripts/v15_value_paths.py, v15_crossdomain.py, v15_figures.py | value_paths.json: junyi.*.paths, connectivity |
| Supplementary S17 to S22 | as above | as above |
