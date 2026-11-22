# Map of release 4.10.0 to the eighth-round manuscript

| Manuscript item | What changed | Source |
|---|---|---|
| Algorithm 2, step 3 | Where the no-cut drop exceeds 0.05, a cut is also flagged when its own drop exceeds twice its seed SD and 0.01; flagged when either reading flags it. "Or fix a unit first" removed | `fuzzykg/audit.py` (`cut_audit`), `scripts/v13_absolute_reading.py` |
| Algorithm 2, step 4 and Output | A grouping counts only if Pi_F (best of three seeds) matches it above the 95th percentile of 20 row permutations of mu_E; then an undecided cut is cut (no loss within 2 seed SDs) or flagged; otherwise it stays undecided. Output reports step 4 for undecided cuts | `fuzzykg/audit.py` (`grouping_reference`, `step4_no_loss`) |
| Section 8.3 | Under x^3 the lower-quartile cut flags on 3 of 5 transition graphs with the absolute reading (0 on the relative rule); the Junyi check rejects the rank-coded drop only; lower-quartile range on the sparse graphs 4 to 7 (mu_E 4, x^3 5, 1-(1-x)^3 6, x^(1/3) 7). Junyi co-practice grouping fails the permutation reference (0.060 against mean 0.065); Junyi transition passes (0.685, above all 20) | `results/v13/absolute_reading.json`, `results/junyi_topic_partition_check.json` |
| Section 9.2 | The 0.05 threshold was set after the drops under x^3 had been seen; largest no-cut drop on mu_E 0.038 | `results/v11/flag_rules.json` |
| Section 9.1, Table 6 D4 | Product 0.426 to 0.567 and hop distance 0.425 to 0.565 on the 1,954 pairs, tau_E 0.2 to 0.1 | `results/junyi_pruning_cross.json` |
| Supplementary Section S3 | Paragraph on the absolute reading and the permutation reference | `results/v13/absolute_reading.json` |
