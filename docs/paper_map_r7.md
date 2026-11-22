# Map of release 4.9.0 to the seventh-round manuscript

| Manuscript item | What changed | Source |
|---|---|---|
| Algorithm 2, step 3 and Output | Unit criterion (meaningful: count share, released or calibrated score; arbitrary: builder-weighted combination); arbitrary unit: cut only where both drops pass, undecided where they disagree; warning when the no-cut drop exceeds 0.05; Output lists both drops, the unit and the decision | `fuzzykg/audit.py` (`decide`, `UNIT_RULE`), `scripts/v12_unit_rule.py` |
| Section 6.3 | Why the transition unit counts as meaningful: the minimum returns one of its two shares | text only |
| Section 8.3, Table 4, Table 6 (D2, D3, D5), Introduction, Section 9.1, Conclusion | Co-practice decisions read by the unit rule: lower quartile 6 cut; median 4 cut, 2 undecided; upper quartile 5 flagged, 1 undecided. Sparse lower quartile: 4 of 7 on mu_E, 4 to 7 under the rescalings the Junyi check does not reject | `results/v12/unit_rule.json` |
| Table 4 caption | Cells near a threshold; the third column uses ten seeds | `results/v12/unit_rule.json`, `results/v11/flag_rules.json` |
| Section 9.2, Table S15 | Step 3 at the settings of Algorithm 2: 141.6 s; steps 1, 3 and 5: 146 s | `results/v12/step3_cost_junyi_copractice.json` |
| Supplementary Section S3 | Decisions by unit, rank-coded grid, near-threshold cells, scope of the Junyi interval | `results/v12/unit_rule.json` |
