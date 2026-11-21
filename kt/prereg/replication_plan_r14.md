# Analysis plan: confirmatory replication on two corpora not used to design the probe

Written 2026-11-21 (UTC+7), before any model was trained on either corpus. Its SHA-256 checksum is recorded in
`prereg/replication_plan_r14.sha256` and in the README of release 3.11.0 at the time the runs were queued; the
runs start only after the checksum is written. Nothing in this plan is changed after the runs; any deviation is
reported as such in the manuscript.

## Corpora

1. **XES3G5M** (Hugging Face `Atomi/XES3G5M_interaction_sequences`), prepared by `src/prep_r14.py xes3g5m`.
2. **Eedi tasks 1-2** (NeurIPS 2020 Education Challenge, `Lucy9999/NIPS34`), prepared by `src/prep_r14.py eedi12`.
   It comes from the same platform and subject tree as the corpus Eedi (tasks 3-4) of the manuscript, with its own
   questions and learner identifiers; it is out of the design set, not a different platform.

Preprocessing, split, subsampling, graph construction, model, budget and evaluation are those of the manuscript,
fixed before this plan: at most 6,000 learners, first 200 interactions, 70/10/20 split by a hash of the learner,
10% of the training learners, five subsamples (s = 0..4), update budget (at most 2,000 updates), one training seed per
subsample, CPU with one thread.

## Runs (`src/r14_controls.py`)

On each of the 10 units (2 corpora x 5 subsamples): `empty`, `support`, `support_emb`, `rvec_emb`, `st_empty`,
`st_er`, `st_dp`, `st_support` (80 runs).

## Outcome

Test AUC on first attempts at a KC (primary) and on all predictions (secondary). For each comparison, the paired
difference per unit.

## Hypotheses (first attempts; directions taken from the manuscript's ten corpora)

- **H1** (graph gain): `support` - `empty` > 0.
- **H2** (graph-free code reproduces the embedding-refinement gain): `rvec_emb` - `empty` > 0, and `support_emb` -
  `rvec_emb` lies within the equivalence bounds of +/- delta.
- **H3** (edges matter where the graph routes evidence): `st_support` - `st_dp` > 0.
- **H4** (a random sparse graph does not help in the state channel): `st_er` - `st_empty` < +delta.

delta = 0.002 AUC, the smallest effect of interest fixed for the whole revision: half the primary first-attempt
gain at 10% under the update budget (+0.0043) and half the seed standard deviation of the support-minus-control
difference (0.0041).

## Estimation and decision rules

- Summary: mean over the two corpora of the corpus means (each corpus mean over its five subsamples).
- Interval: stratified bootstrap, subsamples resampled with replacement within each corpus, 10,000 draws, seed 2026.
- H1, H3 supported if the 95% bootstrap interval lies above 0. H4 supported if the upper bound of the 90% interval
  lies below +delta. H2 supported if the 95% interval of `rvec_emb` - `empty` lies above 0 and the 90% interval of
  `support_emb` - `rvec_emb` lies inside (-delta, +delta) (two one-sided tests at alpha = 0.05).
- Each hypothesis is reported as supported, contradicted (interval wholly on the other side of 0, or for H2/H4 the
  90% interval wholly outside the bounds), or not resolved. Per-corpus means and the number of units with a positive
  difference are reported alongside. No multiplicity adjustment across the four hypotheses; all four are reported.
- No run is excluded. A run that fails is rerun once with the same arguments; if it fails again the unit is dropped
  from every comparison and this is reported.
