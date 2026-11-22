# Map of release 4.11.0 to the ninth-round manuscript

| Manuscript item | What changed | Source |
|---|---|---|
| Section 3.5 | Definitions of a dense and a sparse graph (reciprocity, mean out-degree, share of ordered pairs) and of graded prerequisite evidence (a ranking measured by the Spearman correlation with the crowd score and the share of knowledge-map links oriented correctly; not a calibrated probability or a causal effect) | Table 1, Section 8.1 |
| Section 7 (data) | The crowd score of a pair is the mean of its ratings as released, training and testing files pooled, no rater removed | `data/loaders.py` (`load_junyi_expert`), `scripts/junyi_expert_validity.py` |
| Algorithm 2, step 3 | Run the step at the resolution at which the partition will be used (1 in the paper) | `fuzzykg/audit.py` (`gamma`), `scripts/fkg_audit.py --gamma` |
| Algorithm 2, step 4 | "95th percentile of 20 row permutations" stated as above at least 19 of the 20, each the best of three seeds; the cut comparison uses the mean and SD over the 20 seeds of step 3; the reference is Pi_F, not the support partition | `fuzzykg/audit.py` (`grouping_reference`, `step4_no_loss`) |
| Section 8.5 | Step 3 at gamma 0.5, 1 and 2 on the thirteen graphs: gamma 1 reproduces all 39 decisions; 9 and 8 change; sparse lower-quartile flags 4, 4, 6 of 7; dense graphs fall into 1 to 2 communities at 0.5 and 49 to 240 at 2; undecided cuts 1, 3, 2 | `scripts/v14_gamma_step3.py`, `results/v14/gamma_step3.json` |
| Section 9.2 | Why step 4 reads against Pi_F while step 3 reads against the support; on Junyi transition the lower-quartile cut loses 0.040 against the support and 0.078 against Pi_F (limit 0.054); exploratory audit (steps 1, 3, 5: 146 s) without Algorithm 1 | `results/v11/flag_rules.json`, `results/v12/step3_cost_junyi_copractice.json` |
| Section 9.3 | The long sentence on ground truth split; decisions of step 3 tied to gamma = 1 (17 of 39 change at 0.5 or 2) | `results/v14/gamma_step3.json` |
| Contributions, Fig. 4, Table 6 | Scope of use; the three states in the caption of Fig. 4; basis of every row of Table 6; tau_E trade-off in D7 | - |
| Supplementary Section S3 | Paragraph "Resolution of step 3" with every changed decision and the community counts | `results/v14/gamma_step3.json` |
