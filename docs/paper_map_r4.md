# Items added in the revision of the KBS manuscript that answers the fourth review (release 4.6.0)

Every other item keeps the source listed in `docs/paper_map_r2.md` and `docs/paper_map_r3.md`.

| Item in the manuscript | Script | Result file |
|---|---|---|
| Section 8.4, "Grade-preserving baselines"; Supplementary Table S13 | `fuzzykg/embeddings.py`, `scripts/v9_grade_preserving_baselines.py` | `results/v9/grade_preserving_baselines.json` |
| Section 8.4, the max-min mechanism on the dense graph; Supplementary Section S16 (Tables S16a, S16b) | `scripts/v9_maxmin_error.py` | `results/v9/maxmin_error.json` |
| Section 8.6, paths on PPI5k and CN15k; Supplementary Table S14a | `scripts/v9_cross_domain.py` | `results/v9/cross_domain_paths.json` |
| Section 8.6, percolation of the PPI5k cuts | `scripts/v9_ppi_percolation.py` | `results/v9/ppi5k_percolation.json` |
| Section 8.6, partitions on PPI5k; Supplementary Table S14b | `scripts/v9_audit_cost.py ppi5k --draws 50 --power-draws 10 --graph-seeds 2 --floor-draws 10` (steps 2d to 3 also by `scripts/v9_ppi_parallel_steps.py`, run on a second core) | `results/v9/audit_cost_ppi5k.json` |
| Section 9.2, cost of the protocol; Supplementary Table S15 | `scripts/v9_audit_cost.py junyi_copractice` | `results/v9/audit_cost_junyi_copractice.json` |
| Fig. 4 | `scripts/figure4.py` | `docs/figure4.png`, `docs/figure4.pdf` |
| Table 7 (decision guide) | assembled from the results above and Sections 8.2 to 8.4 | – |
