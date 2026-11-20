# Where each item of the KBS manuscript (third revision, thirteen graphs) comes from (release 4.5.0)

This map follows the numbering of the manuscript on thirteen graphs from four corpora and of its Supplementary material S1 to S12. `docs/paper_map_r2.md` maps the second-revision version (release 4.3.0) and `docs/paper_map.md` the longer version on twenty-three graphs (release 4.2.0).

| Manuscript item | Code | Results |
|---|---|---|
| Fig. 1 (design of the study), full width, text 7.6 to 10 pt at 190 mm | `scripts/figure1.py` | `docs/figure1.png`, `docs/figure1.pdf` |
| Lemma 1 (inherited properties), Claim 1 | `fuzzykg/leiden.py`, `scripts/v6_connectivity.py` | `results/v6/connectivity.json` |
| Lemma 2 (level-set modularity as a fuzzy-graph modularity; rank-coded Q_F drop) | `scripts/check_levelset_identity.py`, `scripts/rank_drop_audit.py` | `results/levelset_identity_check.json`, `results/rank_drop_*` |
| Proposition 1 | `scripts/verify_propositions.py` | |
| Table 1 (graphs) | `scripts/build_*`, `scripts/graph_stats.py` | `data/graphs/`, `results/graph_stats.json` |
| Table 2, Fig. 2, Section 8.1 | `scripts/cost_index_validity*.py`, `scripts/nonnested_*.py`, `scripts/rank_drop_summary.py` | `results/cost_index_validity*.json`, `results/nonnested_*.json` |
| Table 3 (Algorithm 1; Table S3h) | `scripts/calibrated_null_test.py`, `scripts/v6_floor_uncertainty.py`, `scripts/v8_floor_extra.py` (Junyi co-practice), `scripts/v6_power_fine.py` | `results/calibrated_null_test.json`, `results/v6/floor_uncertainty.json`, `results/v8/floor_extra.json`, `results/v6/power_fine.json` |
| Section 8.2, transition construction (Table S6c; floors with ten planted graphs) | `scripts/v6_transition_counts.py`, `scripts/v6_transition_sensitivity.py`, `scripts/v8_floor_extra.py junyi:<variant>` | `results/v6/transition_sensitivity*.json`, `results/v5/conjunction.json`, `results/v8/floor_extra_variants.json` |
| Section 8.2, weighted SBM (Table S11a) | `scripts/v5_weighted_sbm.py` (`SBM_NULL=100 SBM_NULL_INIT=5`; 20 refits for the transition graphs), `scripts/v6_sbm_where.py` | `results/v8/weighted_sbm_100.json`, `results/v8/weighted_sbm_50_full.json`, `results/v6/weighted_sbm_fair.json`, `results/v8/sbm_where.json`, `results/v6/sbm_where.json` |
| Section 8.2, nested SBM (Table S11b; corrected covariate) and clipping check | `scripts/v5_weighted_sbm.py` (`SBM_NESTED=1`; `SBM_CLIP=0.01`) | `results/v8/weighted_sbm_nested.json`, `results/v8/weighted_sbm_clip.json` |
| Section 8.2, orientation and its learner bootstrap | `scripts/v5_junyi_followup.py` | `results/v5/junyi_followup.json` |
| Table 4, Section 8.3 (Table S5d: seed dispersion of the Q_F drop, Algorithm 2 step 3) | `scripts/cost_decomposition.py`, `scripts/real_graph_tables.py`, `scripts/v8_qf_drop_seeds.py` | `results/cost_decomposition.json`, `results/real_graph_tables.json`, `results/v8/qf_drop_seeds.json` |
| Table 5, Fig. 3b, Section 8.4 (expert criteria; cross-fitted threshold, Table S7g) | `scripts/junyi_expert_cluster.py`, `scripts/junyi_pruning_fixed_pairs.py`, `scripts/junyi_unsuggested.py`, `scripts/v5_junyi_followup.py` | `results/junyi_*.json`, `results/v5/junyi_followup.json` |
| Table 6, Fig. 3a (next concept) | `scripts/heldout_transitions.py` | `results/heldout_*.json` |
| Section 7, MOOCCubeX log (Section S12) | `scripts/v6_mooccubex_scan.py`, `scripts/v6_mooccubex_log.py` | `results/v6/mooccubex_log.json` |
| Algorithm 2 | `fuzzykg/audit.py`, `scripts/fkg_audit.py` | `results/v4/audits/` |
