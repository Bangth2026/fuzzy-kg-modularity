# Map of releases 4.8.0 and 4.8.1 to the sixth-round manuscript

| Manuscript item | Script | Result file |
|---|---|---|
| Table 4, fourth and last columns; Section 8.3 decisions; Introduction ("Three design decisions"); Table 6 rows D2, D3, D5 | `scripts/v11_cut_audit_rescale.py`, `scripts/v11_flag_rules.py` | `results/v11/cut_audit_rescale_*.json`, `results/v11/flag_rules.json` |
| Supplementary Table S3e (Q_F drop and rank-coded drop, 20 seeds) | as above | as above |
| Supplementary Table S3f (rescalings, seed halves, rule sensitivity, Junyi topic agreement) | as above | `results/v11/flag_rules.json` (`graphs`, `grid`, `halves`, `ari`) |
| Supplementary Table S3c (recovery thresholds 0.3 / 0.5 / 0.7 with ten planted graphs) | read from `by_p_in` of the floor runs | `results/v6/floor_uncertainty.json`, `results/v8/floor_extra.json` |
| Supplementary Section S13, training objective; Section 8.4 (both Junyi graphs) | `scripts/v11_baselines_excluded.py` | `results/v11/baselines_excluded.json` |
| Supplementary Table S15, refined-tilt rows; Section 9.2 (2.4 h) | `scripts/v11_refined_tilt_cost.py` | `results/v11/refined_tilt_cost.json` |
| Supplementary Table S15, ten-graph floor row | timed by `scripts/v8_floor_extra.py` | `results/v8/floor_extra.json` (`time_s`) |
| Section 8.4, SD of widest-path strengths (0.027) | `scripts/v9_maxmin_error.py` | `results/v9/maxmin_error.json` (`widest.sd_widest`, `widest.sd_direct`) |

Supplementary tables S3d to S3j were relabelled in order of appearance: S3h (floor) is now S3d, S3i is S3e, S3f is new, S3d is S3g, S3e is S3h, S3f is S3i, S3g is S3j.
