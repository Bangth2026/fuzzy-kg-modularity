# Script index

Scripts keep the name of the release that introduced them (prefix `v4_` to `v13_`), because earlier releases, the Zenodo archives and the manuscripts cite those paths. This index gives the function of each script from its own docstring; `docs/paper_map*.md` map tables and figures to scripts.

| Script | Prefix | What it does |
|---|---|---|
| `benchmark_binarisation.py` | - | Planted-partition benchmark for the binarise-then-partition shortcut (Table 7). |
| `benchmark_levelset.py` | - | Level-set multislice modularity against Fuzzy Leiden and the single-cut |
| `build_assistments_graph.py` | - | Section 7.1 end to end on an ASSISTments release (Table 3). |
| `build_junyi_graph.py` | - | Build the two Junyi Academy fuzzy graphs (co-practice, Section 7.1; order-preserving |
| `build_mooccubex_graph.py` | - | MOOCCubeX-CS prerequisite graph (Section 8, Table 2 input). |
| `calibrated_null_test.py` | - | Algorithm 1 (calibrated null test) on the real graphs (Section 8.1, Table 6). |
| `check_levelset_identity.py` | - | Numerical check of the finite identity of Lemma 2 (Section 5.4). |
| `composition_real.py` | - | Composition on the real graphs (Section 8.5): per-source Kendall tau-b against |
| `composition_robustness.py` | - | What the t-norm contributes beyond reachability, and how much of the best |
| `construction_sensitivity.py` | - | Sensitivity of the co-practice construction to its horizon h and minimum support |
| `cost_decomposition.py` | - | Decomposition of the cost index on the real graphs (Section 8.3). |
| `cost_index_validity.py` | - | Does the cost index measure the cost of a cut, and what does it add to free proxies? |
| `cost_index_validity_extended.py` | - | Validity of the cost index, extended (Section 8.1, Table 2). |
| `degree_null_more_draws.py` | - | Degree-preserving null of Algorithm 1 with more draws for one graph. |
| `exact_tnorm_paths.py` | - | Exact best paths under Archimedean t-norms by additive generators (Section 7.3). |
| `external_reference_test.py` | - | Do the memberships favour edges inside an external grouping? (Section 8.2) |
| `figure1.py` | - | Fig. 1 of the KBS manuscript, redrawn for a full-width (190 mm) placement with text >= 7 pt. |
| `figure4.py` | - | Fig. 4 of the KBS manuscript (version 4.6.0): from a learner log to a decision. |
| `fkg_audit.py` | - | Command-line front end of Algorithm 2 (fuzzykg.audit). |
| `gate_baselines.py` | - | Gates on the uncertain state: the cautious collapse against four alternative |
| `graph_stats.py` | - | Graph statistics table (Section 8.1): nodes, edges, density, |
| `heldout_transitions.py` | - | Held-out next-concept prediction as an external criterion for the |
| `hop_vs_tnorm.py` | - | Paired bootstrap (over annotated pairs) of the Spearman correlation of each |
| `junyi_cluster_bootstrap.py` | - | Junyi expert criterion with dependence between pairs taken into account |
| `junyi_expert_cluster.py` | - | Edge-level agreement of the Junyi memberships with the expert score |
| `junyi_expert_validity.py` | - | Validity of the memberships of the Junyi graphs against the expert information |
| `junyi_pruning_cross.py` | - | Graded composition without pruning against hop count after pruning |
| `junyi_pruning_fixed_pairs.py` | - | Pruning threshold of the Junyi transition graph on one fixed set of pairs |
| `junyi_topic_partition_check.py` | - | Does Leiden on Q_F recover the Junyi topics better than crisp Leiden on the support |
| `junyi_unsuggested.py` | - | Robustness of the Junyi transition graph to the platform's own sequencing. |
| `levelset_real.py` | - | Level-set modularity on the real graphs (Section 8.3, Table 9): the |
| `levelset_sensitivity.py` | - | Sensitivity of the level-set objective and the cost index (Section 8.2, |
| `make_junyi_pairs.py` | - | Write the Junyi crowd prerequisite scores as an external-pairs file for scripts/fkg_audit.py |
| `nonnested_binarisation.py` | - | Binarisations of equal size that are not nested (Section 8.1). |
| `nonnested_lfr.py` | - | Equal-size binarisations by four rules on the LFR-style benchmark (Section 8.1). |
| `nonnested_summary.py` | - | Within-graph concordance for the equal-size binarisations of |
| `null_floor_calibration.py` | - | Calibration of the effect-size floor of Algorithm 1 and a seed check on the |
| `null_model_check.py` | - | Algorithm 1 with its null-model test and 'no structure' outcome (Section 5.3). |
| `null_models_extended.py` | - | Four null models for the optimised Q_F at the reference resolution (Table 6). |
| `orness_decision.py` | - | Does the unit invariance of the cost index change a decision? (Section 8.3) |
| `prerequisite_criterion.py` | - | External criteria for path composition that match the prerequisite semantics of |
| `rank_drop_audit.py` | - | The rank-coded Q_F drop, a third audit implied by Lemma 2 (Section 8.1). |
| `rank_drop_summary.py` | - | Summary of the rank-coded Q_F drop (Section 8.1). |
| `real_graph_tables.py` | - | Leiden on Q_F on every real graph at four resolutions (Tables 2-4), with one seed |
| `real_qf_drop.py` | - | Q_F drop of the real-graph cuts, next to the cost index (Section 8.3). |
| `reproduce_section85.py` | - | Section 8.5 / Table 10: interval type-2 state versus the type-1 collapse. |
| `reproduce_table5.py` | - | Section 8.2 / Table 5: Fuzzy Leiden on synthetic fuzzy prerequisite graphs. |
| `reproduce_table9.py` | - | Supplementary Section S1.5 / Table S4: verification of the fuzzification and cut-point sensitivity. |
| `rescaling_decisions.py` | - | Decisions that a rescaling of mu_E changes (Section 8.1). |
| `rescaling_validity.py` | - | Does the price of a cut depend on the unit of the membership scale? (Section 8.1) |
| `run_leiden_grid.py` | - | Tables 2 and 4: Fuzzy Leiden on a saved fuzzy graph over a resolution grid, |
| `scale_invariance_check.py` | - | Does the binarisation conclusion survive strictly increasing rescalings of mu_E? |
| `tnorm_sweep.py` | - | Tables 7 and 8: parametric t-norm families and compensatory alternatives on |
| `v10_cn15k_partition.py` | v10 | The partition half of the audit (Algorithm 2, steps 1 to 3) on CN15k. |
| `v10_gcn_vs_heuristics.py` | v10 | Paired comparison of the grade-preserving models with the best membership-free |
| `v10_link_heuristics.py` | v10 | Membership-free link-prediction heuristics on the hidden MOOCCubeX expert edges. |
| `v11_baselines_excluded.py` | v11 | The grade-preserving baselines of scripts/v9_grade_preserving_baselines.py on the |
| `v11_cut_audit_rescale.py` | v11 | The cut audit of Algorithm 2 (step 3) on the thirteen real graphs, with both |
| `v11_flag_rules.py` | v11 | Flagging rules of Algorithm 2 (step 3) applied to results/v11/cut_audit_rescale_*.json. |
| `v11_refined_tilt_cost.py` | v11 | Measured cost of the refined tilt grid of Table 3 on the Junyi co-practice graph. |
| `v12_step3_cost.py` | v12 | Wall-clock cost of step 3 of Algorithm 2 as the algorithm now specifies it. |
| `v12_unit_rule.py` | v12 | The unit rule of step 3 of Algorithm 2 applied to results/v11/flag_rules.json. |
| `v4_algorithm1.py` | v4 | Algorithm 1 with the structure-matched (degree-corrected) floor. |
| `v4_build_new.py` | v4 | 0: both constructions on five further public corpora (KC level), with the |
| `v4_corpus_stats.py` | v4 | Descriptive statistics of every log after the stated preprocessing (learners, |
| `v4_junyi_labels.py` | v4 | Quality of the Junyi crowd prerequisite labels and what it does to the |
| `v4_nextconcept_baselines.py` | v4 | Held-out next-concept prediction with sequence baselines (reply to the |
| `v4_protocol_external.py` | v4 | Evaluation of the cut audit of Algorithm 2 (step 3) against external groupings, |
| `v4_scaling.py` | v4 | Run time and peak memory of every step of Algorithms 1 and 2 against the |
| `v4_sensitivity.py` | v4 | Sensitivity of the findings to the construction conventions. |
| `v4_summarise.py` | v4 | Collects the v4 results into the tables of the revised manuscript |
| `v4_table4_new.py` | v4 | Table 4 rows for the ten graphs of the added corpora: retained mass and mean |
| `v5_audit_agreement.py` | v5 | Agreement of the cut audit (Algorithm 2, step 3) with ARI loss against independent groupings. |
| `v5_conjunction.py` | v5 | Which edges does a compensatory conjunction add to the Junyi transition graph? |
| `v5_figure2.py` | v5 | Figure 2 of the KBS manuscript (release 4.2.0). |
| `v5_gamma_audit.py` | v5 | Sensitivity of the cut audit (Algorithm 2, step 3) to the resolution gamma. |
| `v5_junyi_followup.py` | v5 | Follow-up analyses on Junyi (release 4.2.0). |
| `v5_weighted_sbm.py` | v5 | Weighted stochastic block model check: do the memberships carry block information beyond topology? |
| `v6_connectivity.py` | v6 | How many communities the Claim 1 check covers. |
| `v6_floor_uncertainty.py` | v6 | Sampling uncertainty of the decision floor of Algorithm 1 (step 3). |
| `v6_mooccubex_log.py` | v6 | How much of the MOOCCubeX exercise log falls on Computer-Science concepts. |
| `v6_mooccubex_scan.py` | v6 | Resumable full pass over MOOCCubeX relations/user-problem.json by HTTP range requests. |
| `v6_power_fine.py` | v6 | Finer tilt grid for the power of the weights-only nulls (Algorithm 1, step 5). |
| `v6_sbm_where.py` | v6 | Where does the weight channel of the weighted SBM act? |
| `v6_transition_counts.py` | v6 | Run-to-run move counts T(a -> b) of the Junyi log, for every ordered pair. |
| `v6_transition_sensitivity.py` | v6 | Robustness of the community findings to the transition construction (Junyi). |
| `v7_qf_drop_seed_sd.py` | v7 | Seed dispersion of the Q_F drop on the real graphs (Algorithm 2, step 3). |
| `v7_variant_floor.py` | v7 | Ten-graph decision floor for variants of the Junyi transition construction. |
| `v8_floor_extra.py` | v8 | Decision floor of Algorithm 1 with ten planted graphs where it still rested on two. |
| `v8_qf_drop_seeds.py` | v8 | Seed dispersion of the Q_F drop on the thirteen real graphs (Algorithm 2, step 3). |
| `v9_audit_cost.py` | v9 | Measured cost of the full pre-deployment audit (Algorithm 2) on one graph. |
| `v9_cross_domain.py` | v9 | The path half of the audit on graded knowledge graphs outside education. |
| `v9_grade_preserving_baselines.py` | v9 | Grade-preserving baselines against the expert criteria of Section 8.4. |
| `v9_maxmin_error.py` | v9 | Why max-min erases the graded evidence on the dense Junyi co-practice graph. |
| `v9_merge_ppi_audit.py` | v9 | Merge results/v9/audit_ppi5k_rest.json (scripts/v9_ppi_parallel_steps.py) into |
| `v9_ppi_parallel_steps.py` | v9 | Steps 2d to 3 of Algorithm 2 on PPI5k (global and row weights-only nulls, 50 draws, |
| `v9_ppi_percolation.py` | v9 | Percolation of the alpha-cuts of PPI5k (Section 8.6): share of ordered pairs joined |
| `verify_propositions.py` | - | Numerical verification of Propositions 1-3, Corollary 1, Remark 3 and Fact 1. |
| `weights_null_more_draws.py` | - | Weights-only nulls (global and row) with more draws for one graph. |
| `v14_gamma_step3.py` | v14 | Step 3 of Algorithm 2 at gamma 0.5, 1 and 2 on the thirteen real graphs, with community counts (results/v14/gamma_step3.json). |
| `v13_absolute_reading.py` | v13 | Absolute reading of step 3 where the no-cut drop exceeds 0.05, and the permutation reference of step 4 (results/v13/absolute_reading.json). |
| `v15_checks.py` | v15 | Numerical check of Proposition 1, check of the kappa* search and cost of the unit-robust audit (Supplementary S17). |
| `v15_crossdomain.py` | v15 | Path results and path value of the grades on PPI5k and CN15k (Section 8). |
| `v15_figures.py` | v15 | Figures 1 to 3 of the release 5.0.0 manuscript. |
| `v15_learner_bootstrap.py` | v15 | Stability of the step-3 decisions under a bootstrap of the learners (Section 8.2). |
| `v15_partitions.py` | v15 | Cached Leiden partitions of the 23 graphs reused by every v15 script. |
| `v15_semisynthetic.py` | v15 | Semi-synthetic logs with a known answer: the audit, paths and value of grades (Section 7). |
| `v15_simplex_real.py` | v15 | Unit-robust cut audit (Algorithm 1) on the 23 real graphs (Section 8.2). |
| `v15_summarise.py` | v15 | Collects every number of the release 5.0.0 manuscript into results/v15/summary.json. |
| `v15_value_paths.py` | v15 | Value of grades at three levels, connectivity level and hop-penalised score on the real graphs (Sections 8.1, 8.3). |
