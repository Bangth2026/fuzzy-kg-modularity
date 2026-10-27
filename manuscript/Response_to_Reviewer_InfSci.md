# Response to the referee report

**Manuscript:** What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs built from learner logs (formerly "... strict t-norms ...")

**Author:** Hai Bang Truong, Saigon International University

We thank the referee for a careful, technically precise report. Three of the seven major concerns identified errors in our claims, and we have corrected them rather than argued with them; the report also pointed to the experiment that settles the binarisation question, which we have run. Below, the referee's text is paraphrased in italics, our reply follows, and the location of each change is given. Line numbers refer to the revised manuscript (v2). Four quantities that require re-running the released pipeline on the ASSISTments and MOOCCubeX files are marked **[to be filled before resubmission]** and are listed at the end.

---

## M1. The Corollary 1 gap is not the cost of binarising

*The level-set quantity (9) is normalised by W, not by |E_α|; a constant-membership graph is lossless to binarise yet has a gap γ(1/c − 1)Σ a_κ b_κ; Table 6 already contradicts the interpretation; the shortcut experiment reused the fuzzy-graph γ on sparse level sets, used thresholds no practitioner would choose, one detector and the best of five seeds.*

We agree on every point, including the counterexample, which we have verified numerically and adopted as **Remark 3** (Section 5.4). The reading of the gap as the cost of binarising has been withdrawn from the title-level claims, the abstract, highlights, contributions, Sections 5.4, 10.1 and 11, and from the practitioner recommendation. The phrase "zero only in the crisp case" is removed and the equality condition of Proposition 2 is stated as it stands.

Changes:

1. **Remark 3** (Section 5.4) states the counterexample with the closed form of the gap and explains what the gap does measure (how far memberships sit below one under global normalisation). The released script checks the identity on a planted graph with all memberships equal to c ∈ {0.25, 0.5, 0.75}: Q_F equals the crisp modularity of the same partition, binarising is lossless, and the gap equals γ(1/c − 1)Σ a_κ b_κ to fifteen decimals (Section 9.1).
2. Section 5.4 now says explicitly that (9) is not the modularity of G_α, with the r = |E_α|/W weighting the referee gave.
3. **The cost is defined operationally** as loss(α) = Q_F(Π_fuzzy) − max_γ′ Q_F(Π_α(γ′)) with γ′ refitted per level set, together with NMI/ARI against the planted partition and NMI between Π_α and Π_fuzzy (Section 5.4, end).
4. **Planted-partition benchmark with ground truth (new Table 6).** Directed planted partitions, 1000 nodes, 20 communities, 8 out-edges per node (70% internal), Beta memberships with mean ∈ {0.3, 0.5, 0.7, 0.9} and SD ∈ {0.05, 0.15, 0.25} set independently, with and without community-correlated memberships (shift 0 / 0.5), three graph seeds; cuts at absolute α ∈ {0.4, 0.6, 0.8} and at membership quartiles (the scale-free variant the referee asked for); Fuzzy Leiden and crisp Leiden each scanned over γ′ ∈ {0.25, 0.5, 1, 2, 4} × 3 seeds, best Q_F on the fuzzy graph kept. Weighted Leiden on the memberships is Fuzzy Leiden by definition of Q_F. Result: loss and NMI are governed by the fuzzy mass the cut retains (Spearman −0.81 / +0.80 over 420 cut cells); the membership spread has no detectable effect (−0.03); the Corollary 1 gap is a poor predictor (0.43). The regime statement is in retained mass: free above ~60%, a few hundredths between 40 and 60%, destructive below 40% at any resolution. Our earlier claim that "spread decides" is withdrawn in the text.
5. The real-graph rows (now Table 8) are retained with an explicit caveat that they were computed at a single resolution, and the released script now refits γ per level set and reports NMI(Π_α, Π_fuzzy) on every graph. **[to be filled before resubmission: Table 8 with refitted γ, which also answers the referee's question 2 on α = 0.6 and 0.8.]**
6. The "tightness" of the Cauchy–Schwarz bound is now explained as expected on near-reciprocal graphs (f_κ ≈ g_κ) rather than reported as a finding (Sections 5.4 and 9.1).

## M2. The resolution result is mislabelled; Algorithm 1 cannot fail

*The statement is a sufficient condition for balanced splits, not a lower bound; it is the standard split/merge gain; the numbering is inconsistent; Algorithm 1 is true by construction and returns near-singleton partitions on the KC-level graphs; no external validation; are the 28 communities of the 2009 item-level graph components or problem sets?*

Accepted. Changes:

1. **Proposition 1** (Section 5.3) is now stated for general bipartitions: Q_F(B) beats the trivial partition iff γ > θ(B) = (c(B)/W)/(a₁b₂ + a₂b₁); 2c/W is the balanced-split special case and is identified as an *upper* bound on the onset. The proof notes that the denominator is maximised by the balanced split. The numbering is consistent throughout (the stray "Lemma 1(i)" label is gone), and the two cross-reference errors the referee caught in Table 4's caption and Section 9.1 (Lemma 1(iv), Lemma 1(ii)) are corrected.
2. The identity is attributed to the local-moving gain of Louvain/Leiden and to the resolution-limit literature (Reichardt and Bornholdt [45]; Arenas et al. [46]; Traag et al. [47]; Newman's weighted modularity [44]); the sentence claiming we had not found the bound is removed (Section 2.1).
3. The referee's data point is used: the 2017 item-level graph returns 16 communities at γ = 0.5 below 2c/W = 0.531 because the partition is not a balanced bipartition (Sections 5.3 and 9.1). The released code reports θ(B) for the spectral bisection and for every community-vs-rest split of the accepted partition (`bipartition_threshold_general`).
4. **Algorithm 1 revised** with a degree-preserving null model (igraph rewiring preserving in/out degrees, memberships permuted over edges, Fuzzy Leiden at the same γ and seeds, z-score of the best Q_F) and a "no structure detected" outcome (steps 8–11). We state plainly that the earlier version could not fail and that the KC-level outcome is a second degenerate output, not an informative one.
5. **New Table 7** reports the revised algorithm where its behaviour can be checked: planted p_in = 0.8 → "structure" (z ≫ 10³); same construction with p_in = 1/k → the seeding loop climbs to γ₁ ≈ 109 and 582 communities, null test returns "no structure detected" (z = 0.6); a dense co-practice-like graph (110 nodes, 50 edges/node) → 109 communities, Q_F = −0.041, "no structure detected" (z = −0.1). **[to be filled before resubmission: the same test on the six real graphs; `scripts/null_model_check.py data/graphs/*_edges.csv`.]**
6. The 28-community question: we could not settle it from Q_F alone. The released code now reports weakly connected components and the problem-set partition next to every partition, and Section 10.3 lists the comparison against that reference partition as an open check rather than claiming "prerequisite structure".
7. Comparison with resolution-profile, consensus and resolution-limit-free methods: not performed in this revision; Section 10.3 states this explicitly as a limitation and cites the methods.

## M3. The strict t-norm section: incorrect complexity claim and weak measurement

*Best paths under every family used are shortest-path problems via additive generators; "strict" is misused; Fact 1 and 7.2 are textbook; pooled Spearman is dominated by hop length and ties; the causal claim is not identified; an ordinal-invariance argument is missing.*

The complexity claim was wrong and is withdrawn everywhere (abstract, contributions, Sections 9.3, 10.1, 11). We thank the referee for the generator argument, which we have adopted and extended.

1. **New Section 7.3** states the generator reduction: S_T(π) = g^(−1)(Σ g(μ_e)), best path = shortest path on non-negative lengths, Dijkstra O(|E| log |C|), L-round Bellman–Ford exact under a hop limit (non-negative lengths ⇒ the minimum over walks is attained by a simple path). **Table 13** lists the generators of product, Łukasiewicz, Yager and Schweizer–Sklar with their g(0) and kind. The released code implements `generator()` and `generator_best_paths()`; the earlier enumeration agrees with it to 10⁻¹⁵ on every family and every target checked.
2. **Cost recomputed** (Section 9.3): on 371-node graphs at 1.5/4/8 edges per node, Dijkstra on generator lengths costs 0.73×/0.39×/0.26× the max-min sweep; the enumeration grows from 0.3× to 15×. The "350×" is stated to have measured our enumeration.
3. **Terminology:** "strict" replaced by "non-idempotent" in the title and abstract and by "Archimedean" where the generator argument applies; Section 7.2 now distinguishes strict (product, Schweizer–Sklar p < 0) from nilpotent (Łukasiewicz, Yager, Schweizer–Sklar p > 0).
4. **Ordinal-invariance argument added as Proposition 3** (Section 7.3) with proof: max-min best paths and target orderings are invariant under every strictly increasing rescaling of μ_E; for every continuous Archimedean t-norm a graph and a rescaling exist that change the best path. Verified on the synthetic graph under ten random increasing piecewise-linear maps from twenty sources: Kendall τ = 1.000 for max-min on every source and map; mean 0.97 (min 0.33) product, 0.90 (0.33) Łukasiewicz, 0.88/0.98 Yager p = 0.5/2, 0.99 Schweizer–Sklar p = −2. This is the controlled experiment with fixed topology and transformed memberships the referee asked for, and it replaces the mean-vs-spread causal claim, which is withdrawn (Section 9.3, "First").
5. Fact 1 and Section 7.2 are kept as framing only; the contribution list no longer mentions them.
6. **Rank statistic:** the text now states that the pooled ρ is partly a hop-length and tie artefact and that the per-source Kendall τ-b is the relevant quantity; the released script computes per-source τ-b with bootstrap intervals. **[to be filled before resubmission: Table 11 recomputed without the four-hop limit and with per-source τ-b and bootstrap CIs; `scripts/exact_tnorm_paths.py data/graphs/assist2009_item_edges.csv`.]**
7. The dioid/algebraic-path literature is cited (Gondran and Minoux [49], Section 2.3).

## M4. Interval type-2 section: baseline, novelty and evidence

*The cautious collapse is the author's own construction; obvious baselines (i)–(iv) are missing; "not embedded" is internal; the headline numbers are grid properties; σ is assumed to be a standard deviation; η = 1 has no probabilistic content; no evidence the representation helps a decision.*

We accept all of this and have followed the referee's second option: Section 6 is reduced to a methodological remark and the headline numbers (7.3%, 0.075, 2.8×) are removed from the abstract, highlights, contributions and conclusion.

1. Section 6.3 now says the collapse is our own construction, that we know of no published pipeline using it, and that the earlier "common practice" claim was wrong (also Section 1).
2. The degeneration is presented as a defect of that construction, with the referee's three-line derivation, not of type-1 representations.
3. **Baselines (i)–(iv) implemented and compared** (new Table 13, Section 9.5): point-estimate gate, interval-image type-1 gate (which is the IT2 "pass" rule), probability-of-ZPD under a normal predictive distribution thresholded at ½, expected membership over the interval. Finding stated plainly: the collapse and the interval-image gate admit the same states up to ησ = 0.05; the IT2 form does not avoid rejection, it relabels part of it as indeterminate, "and that is the whole of what the type-2 form adds"; the probability and expected-membership gates keep admitting to ησ = 0.20. The table also withdraws the earlier Table 12, which normalised the bound vectors separately before gating (a procedure under which pass and fail can both fire; the raw-bounds gate is used now).
4. Embeddedness is kept as a cheap diagnostic with an explicit statement that no harm to decisions is claimed.
5. η = 1: the Chebyshev coverage at the core is 1 − η⁻² = 0; stated in Section 6.3, Table 1's source for η changed from "T" to "P (core of π_C; no coverage)".
6. σ: the 27% relative standard error at R = 8 and the calibration caveat (Ovadia et al. [50]) are stated; Lemma 2's semantics are said to presuppose a σ never checked here.
7. Real estimator outputs, calibration, selective-prediction curves on held-out outcomes: not performed. Section 10.2 states this as the experiment that would settle Section 6 and declares it out of scope for this paper; the representation is kept in the released code.

## M5. Claims exceed the evidence; the third question is unanswered

Agreed. Changes: the third question is now answered with ground truth for binarisation only (planted partitions), and the Introduction, Section 10.2 and the Conclusion say that for the other two shortcuts the evidence is proxy-based. The practitioner recommendations are rewritten as measured statements ("report retained mass", "test against a null model", "use max-min unless the scale is cardinal", "treat the cautious collapse as a choice with a measurable failure width"). Section 10.3 adds "The real graphs have no reference structure" and lists the two checks the code is prepared for (MOOCCubeX log vs expert pairs; problem-set reference partition). The "prerequisite reading holds at item level" sentence is downgraded to a question pending the component/problem-set comparison. Table 9's density mismatch with the KC-level regime is stated (Section 9.2). The "tiny curated graph" caveat is added to Table 10's discussion. The claim "the KC-level outcome is informative" is withdrawn (M2).

## M6. Coherence, depth and related work

We have deepened the first thread as the referee suggested (ground-truth benchmark, general threshold, null-model test) and reduced the second to a remark; the third now carries a proven proposition (ordinal invariance) in place of an empirical sweep plus an incorrect complexity claim. Related work added: Newman 2004 [44], Reichardt–Bornholdt 2006 [45], Arenas et al. 2008 [46], Traag et al. 2011 [47], Lancichinetti–Fortunato 2012 [48], Gondran–Minoux [49], Ovadia et al. 2019 [50]. We acknowledge that resolution-selection baselines, interval-valued edges by bootstrap over learners (referee's §4.5) and an outcome-level estimator study remain open; the first two are named in Section 10.3 and the Conclusion as the directions that follow.

## M7. Presentation and internal consistency

- The three Vietnamese draft notes are resolved: τ_E, R, τ_path were hand-set and are now labelled P with a sentence in Section 8.4; the 79.7 ms figure is stated to be a forward-pass timing on an initialised network of the stated architecture (Section 9.6); reference [2] is corrected to the published AAAI-26 version (title, authors, volume, pages, DOI).
- Proposition 1 / Lemma 1(i) labelling unified; Table 4 caption → Lemma 1(iv); Section 9.1 boundedness → Lemma 1(ii).
- "Zero only in the crisp case" removed (abstract, Section 5.4, Conclusion).
- Lemma 1(iv) and (i)/(v): we have not yet rewritten the lemma to specify refined vs unrefined partitions or to re-check the undirected assumptions of [7] under the Leicht–Newman null model. **[author action before resubmission: amend Lemma 1 statements accordingly.]**
- Duplicate Figure 1 caption removed; the "within 0.004 … within 0.004" sentence rewritten.
- Abstract rewritten around claim, evidence and limitation; most figures removed from it.

## Minor points

- §5.4 "disperses": rephrased after the correction.
- §9.1 "both propositions hold": now "were not violated", described as sanity checks.
- §9.2 aggregation levels: attribution stated as such; the script records the number of levels per run. **[to be filled: the measured number.]**
- §9.4 argmax vs gate: stated; the script reports gate flips as well as band flips.
- Table 12 (now 13) "agrees with Nie–Tan" column removed; agreement is not accuracy.
- §9.5 cost: "negligible", 2.8× removed from abstract and contributions.
- §8.1 Eq. (14) denominator: the reasoning (both orders counted so that prec is a share, not a rate) remains to be stated and tested on a synthetic log with known order. **[author action.]**
- §8.2 R = 8 plateau: not shown; R is now labelled P and the sensitivity to R is listed as unmeasured.
- "Six public fuzzy graphs" → "six graphs from four corpora" (Highlights, Abstract, Section 9).
- "114 reported values" → "the values listed in the supplement"; the list is to be produced by the verify stage. **[to be filled.]**

## Answers to the referee's questions

1. Yes. The gap equals γ(1/c − 1)Σ a_κ b_κ on a constant-membership graph while binarising is lossless; we verified it numerically and it is now Remark 3. The counterexample does not fail; our interpretation did.
2. On planted partitions with γ refitted per level set the loss at the corresponding retained-mass levels is in Table 6; on the real graphs the refitted values are pending (Table 8, marked).
3. We do not yet know; the code now reports components and problem-set partitions, and the claim is withdrawn until the comparison is run.
4. Yes. The 350-fold and "exponential" statements are withdrawn; on synthetic graphs Dijkstra on generator lengths costs 0.26–0.73× the max-min sweep (Section 9.3).
5. None that we know of; the collapse is our own. On the grid the IT2 gate's pass set equals that of the interval-image gate min_{[lo,hi]} m_par ≥ τ (Table 13).
6. Not measured in this paper; stated as out of scope (Section 10.2).
7. For binarisation: a cut retaining under 40% of the fuzzy mass changes the community partition at any resolution (NMI 0.4–0.7 against the truth on planted graphs). For the other two shortcuts the outcome-level measurement is not available on the released data and is not claimed.

## Items to complete before resubmission (require the public data files)

1. Table 8: binarisation shortcut on the six real graphs with γ refitted per level set and NMI(Π_α, Π_fuzzy).
2. Table 7, last row: null-model z and outcome on the six real graphs.
3. Table 11: recomputed without the hop limit; per-source Kendall τ-b with bootstrap intervals.
4. Components and problem-set partition of every real graph; aggregation levels per run; list of verified values for the supplement.
5. Lemma 1 wording (refined vs unrefined partition; directed null-model assumptions); Eq. (14) denominator rationale.
