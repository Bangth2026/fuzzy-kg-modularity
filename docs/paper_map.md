# Where each item of the KBS manuscript comes from (release 4.2.0)

Release 4.2.0 adds the analyses of the revision (`results/v5/`); every result of releases 4.0.0 and 4.1.0 is unchanged. Supplementary material B is a technical appendix with its own section numbers and tables B1 to B14; Lemma B2 there is Lemma 1 of the main article, and comments in the code follow the numbering of Supplementary material B.

| Main article | Supplementary B | Code | Results |
|---|---|---|---|
| Lemma 1 (level-set modularity as a fuzzy-graph modularity; rank-coded Q_F drop) | Lemma B2, Appendix A.2 | `fuzzykg/audit.py` (`rank_code`), `scripts/check_levelset_identity.py`, `scripts/rank_drop_audit.py` | `results/levelset_identity_check.json`, `results/rank_drop_*` |
| Proposition 1 (ordinal invariance of max-min) | Proposition 1, Appendix A.3 | `scripts/verify_propositions.py` | |
| Algorithm 1 (calibrated null test, both floors) | Section 5.5, Table B3 | `fuzzykg/nulltest.py`, `scripts/calibrated_null_test.py`, `scripts/v4_algorithm1.py` | `results/calibrated_null_test.json`, `results/v4/algorithm1_new.json`, `results/v4/dcfloor.json` |
| Algorithm 2 (pre-deployment audit) | Section 9.2 | `fuzzykg/audit.py`, `scripts/fkg_audit.py` | `results/v4/audits/` |
| Section 3 (orientation of map links, conjunction) | Section 8.9 | `scripts/v5_junyi_followup.py`, `scripts/v5_conjunction.py` | `results/v5/junyi_followup.json`, `results/v5/conjunction.json` |
| Table 1 (data) | Table B1 | `data/loaders.py`, `scripts/build_*`, `scripts/v4_build_new.py`, `scripts/v4_corpus_stats.py` | `data/graphs/`, `results/v4/corpus_stats.json` |
| Table 2 (protocol) | Section 7 | as the rows below | |
| Table 3 (do the audits measure the cost of a cut?) | Table B2, Section 8.1 | `scripts/cost_index_validity.py`, `scripts/cost_index_validity_extended.py`, `scripts/nonnested_binarisation.py`, `scripts/nonnested_lfr.py`, `scripts/rank_drop_summary.py` | `results/cost_index_validity*.json`, `results/nonnested_*.json` |
| Table 4 (Algorithm 1 and the weighted SBM on 23 graphs) | Tables B3 and B13, Sections 8.2 and 8.8 | as Algorithm 1; `scripts/v5_weighted_sbm.py` | as Algorithm 1; `results/v5/weighted_sbm.json` |
| Table 5 (cut audit against independent groupings) | Tables B10, B14 and B14b, Sections 8.6 and 8.9 | `scripts/v4_protocol_external.py`, `scripts/v5_audit_agreement.py`, `scripts/v5_gamma_audit.py` | `results/v4/protocol_external.json`, `results/v5/audit_agreement.json`, `results/v5/gamma_audit.json` |
| Table 6 (composition against Junyi expert labels) | Tables B5 and B14c, Sections 8.4 and 8.9 | `scripts/junyi_expert_cluster.py`, `scripts/junyi_pruning_fixed_pairs.py`, `scripts/junyi_unsuggested.py`, `scripts/v5_junyi_followup.py` | `results/junyi_*.json`, `results/v5/junyi_followup.json` |
| Table 7 (decisions for a knowledge engineer) | Table B12, Section 9.1 | `scripts/v4_summarise.py` | `results/v4/paper_numbers.json` |
| Fig. 2 | – | `scripts/v5_figure2.py` | `docs/figure2.png` |
| Section 8.3 (binarising the real graphs) | Table B4, Section 8.3 | `scripts/cost_decomposition.py`, `scripts/v4_table4_new.py` | `results/cost_decomposition.json`, `results/v4/table4_new.json` |
| Section 8.5 (next concept, sensitivity, cost) | Tables B6 to B9 and B11 | `scripts/heldout_transitions.py`, `scripts/v4_nextconcept_baselines.py`, `scripts/v4_sensitivity.py`, `scripts/v4_scaling.py` | `results/heldout_*.json`, `results/v4/nextconcept_*.json`, `results/v4/sensitivity.json`, `results/v4/scaling.json` |
