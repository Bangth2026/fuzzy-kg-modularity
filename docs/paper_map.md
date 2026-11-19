# Where each item of the KBS manuscript comes from (release 4.1.0)

Release 4.1.0 matches the version of manuscript (A) shortened to the length that Knowledge-Based Systems prefers (20 double-spaced pages including tables and figures). The computations are those of release 4.0.0, and no number changed. The main article keeps the central results. Supplementary material B, an extended report, holds the complete derivations, proofs and per-graph results, with its own section, table and lemma numbers. Comments in the code follow the numbering of Supplementary material B.

| Main article | Supplementary B | Code | Results |
|---|---|---|---|
| Lemma 1 (level-set modularity as a fuzzy-graph modularity; rank-coded Q_F drop) | Lemma 2, Appendix A.2 | `fuzzykg/audit.py` (`rank_code`), `scripts/check_levelset_identity.py`, `scripts/rank_drop_audit.py` | `results/levelset_identity_check.json`, `results/rank_drop_*` |
| Proposition 1 (ordinal invariance of max-min) | Proposition 1, Appendix A.3 | `scripts/verify_propositions.py` | |
| Algorithm 1 (calibrated null test, both floors) | Section 5.5, Table 3 | `fuzzykg/nulltest.py`, `scripts/calibrated_null_test.py`, `scripts/v4_algorithm1.py` | `results/calibrated_null_test.json`, `results/v4/algorithm1_new.json`, `results/v4/dcfloor.json` |
| Algorithm 2 (pre-deployment audit) | Section 9.2 | `fuzzykg/audit.py`, `scripts/fkg_audit.py` | `results/v4/audits/` |
| Table 1 (do the audits measure the cost of a cut?) | Table 2, Section 8.1 | `scripts/cost_index_validity.py`, `scripts/cost_index_validity_extended.py`, `scripts/nonnested_binarisation.py`, `scripts/nonnested_lfr.py`, `scripts/rank_drop_summary.py` | `results/cost_index_validity*.json`, `results/nonnested_*.json` |
| Table 2 (cut audit against independent groupings) | Table 10, Section 8.6 | `scripts/v4_protocol_external.py` | `results/v4/protocol_external.json` |
| Table 3 (composition against Junyi expert labels) | Table 5, Section 8.4 | `scripts/junyi_expert_cluster.py`, `scripts/junyi_pruning_fixed_pairs.py`, `scripts/junyi_unsuggested.py` | `results/junyi_*.json` |
| Table 4 (decisions for a knowledge engineer) | Table 12, Section 9.1 | `scripts/v4_summarise.py` | `results/v4/paper_numbers.json` |
| Section 8.2 (what log-built grades carry) | Tables 3 and S3d, Section 8.2 | as Algorithm 1; `scripts/junyi_topic_partition_check.py` | `results/junyi_topic_partition_check.json` |
| Section 8.3 (binarising the real graphs) | Table 4, Section 8.3 | `scripts/cost_decomposition.py`, `scripts/v4_table4_new.py` | `results/cost_decomposition.json`, `results/v4/table4_new.json` |
| Section 8.5 (next concept, sensitivity, cost) | Tables 6 to 9 and 11 | `scripts/heldout_transitions.py`, `scripts/v4_nextconcept_baselines.py`, `scripts/v4_sensitivity.py`, `scripts/v4_scaling.py` | `results/heldout_*.json`, `results/v4/nextconcept_*.json`, `results/v4/sensitivity.json`, `results/v4/scaling.json` |
| Data (twenty-three graphs) | Table 1, Section 7 | `data/loaders.py`, `scripts/build_*`, `scripts/v4_build_new.py` | `data/graphs/`, `results/v4/corpus_stats.json` |
