# junyi_transition_edges.csv

## Graph

- n_active: 628
- m: 1078
- density: 0.002738
- reciprocity: 0
- mean_out_degree: 1.717
- mu_mean: 0.4003
- mu_sd: 0.1813

## Cuts

Rule: flag a cut whose audit exceeds the no-cut value by > 2 x max(seed SD) (10 seeds).

| cut | alpha | retained | mean out-degree | Q_F drop (SD) | rank-coded drop (SD) | flagged | ARI with grouping |
|---|---|---|---|---|---|---|---|
| support | - | 1.00 | 1.72 (low) | 0.0102 (0.0026) | 0.0180 (0.0030) | - | nan |
| q0.25 | 0.253 | 0.75 | 1.36 (low) | 0.0321 (0.0032) | 0.0195 (0.0023) | yes | nan |
| q0.50 | 0.340 | 0.50 | 1.02 (low) | 0.1267 (0.0047) | 0.1097 (0.0034) | yes | nan |
| q0.75 | 0.500 | 0.25 | 0.74 (low) | 0.3961 (0.0012) | 0.4095 (0.0015) | yes | nan |

## Path inference

| score | rho | reached | minus hop (95% CI) |
|---|---|---|---|
| direct_edge | 0.375 | 0.13 | [-0.131, -0.020] |
| hop_distance | 0.448 | 0.45 | - |
| min | 0.419 | 0.45 | [-0.053, -0.011] |
| product | 0.448 | 0.45 | [-0.010, +0.008] |
| lukasiewicz | 0.415 | 0.17 | [-0.089, +0.021] |
| yager_5 | 0.466 | 0.38 | [-0.011, +0.048] |

Recommendation: hop distance on the support (no operator beats it).
Operators below the direct edge: none.
