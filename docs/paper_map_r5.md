# Items added or changed in the revision of the KBS manuscript that answers the fifth review (release 4.7.0)

Every other item keeps the source listed in `docs/paper_map_r2.md`, `docs/paper_map_r3.md` and `docs/paper_map_r4.md`.

| Item in the manuscript | Script | Result file |
|---|---|---|
| Table 4, fourth column (cuts flagged by the Q_F drop, Algorithm 2 step 3); Section 8.3; Table 7, D2 and D3 | `scripts/v7_qf_drop_seed_sd.py` | `results/v7/qf_drop_seed_sd.json` (Supplementary Table S3i) |
| Section 8.3 and Supplementary Table S3i, rank-coded Q_F drop on MOOCCubeX and Junyi transition | `scripts/v8_qf_drop_seeds.py` | `results/v8/qf_drop_seeds.json` |
| Table 3 and Section 8.2, ten-graph floor of the Junyi co-practice graph, 0.046 [0.007, 0.046] | `scripts/v8_floor_extra.py` | `results/v8/floor_extra.json` (Table S3h) |
| Section 8.2 and Supplementary Table S6c, ten-graph floors of all twelve transition variants | `scripts/v8_floor_extra.py` | `results/v8/floor_extra_variants.json` |
| Section 8.2, weighted SBM with 100 refits (50 on the full MOOCCubeX graph), nested SBM with the covariate, clipping check | `scripts/v5_weighted_sbm.py` | `results/v8/weighted_sbm_100.json`, `results/v8/weighted_sbm_50_full.json`, `results/v8/weighted_sbm_nested.json`, `results/v8/weighted_sbm_clip.json`, `results/v8/sbm_where.json` (Supplementary Section S11) |
| Section 8.4, membership-free heuristics on the hidden MOOCCubeX edges; Supplementary Section S13 | `scripts/v10_link_heuristics.py` | `results/v10/link_heuristics.json` |
| Section 8.4, paired differences GCN / UKGE-style model / product against preferential attachment | `scripts/v10_gcn_vs_heuristics.py` | `results/v10/gcn_vs_heuristics.json` |
| Section 8.6 and Supplementary Section S14, cut audit on CN15k and the cost of Algorithm 1 there | `scripts/v10_cn15k_partition.py` | `results/v10/cn15k_partition.json` |
| Fig. 4 (decision guide D1 to D9) | `scripts/figure4.py` | `docs/figure4.png`, `docs/figure4.pdf` |
