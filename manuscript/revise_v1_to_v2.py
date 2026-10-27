# -*- coding: utf-8 -*-
"""Apply the revision to ms_v1.md -> ms_v2.md.  Every replacement asserts that
its anchor text exists exactly once, so a silent mismatch cannot happen."""
import re, json, sys

s = open("ms_v1.md", encoding="utf-8").read()
R = json.load(open("/home/claude/fuzzy-kg-modularity/results/binarisation.json"))
T = json.load(open("/home/claude/fuzzy-kg-modularity/results/tnorm_paths.json"))
NM = json.load(open("/home/claude/fuzzy-kg-modularity/results/null_model.json"))
GB = json.load(open("/home/claude/fuzzy-kg-modularity/results/gate_baselines.json"))

def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)

def sub(pattern, new, flags=0):
    global s
    s2, n = re.subn(pattern, new, s, flags=flags)
    assert n >= 1, pattern[:80]
    s = s2

# ------------------------------------------------------------------ title
rep("# What binarisation, type-1 collapse and strict t-norms cost on fuzzy knowledge graphs built from learner logs",
    "# What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs built from learner logs")

# ------------------------------------------------------------- highlights
rep("""-   Binarising a fuzzy graph before partitioning loses a bound
    computable from the graph
-   A closed-form resolution threshold explains and repairs degenerate
    Leiden runs
-   Type-1 collapse of an uncertain state rejects every candidate past a
    closed-form width
-   Rank agreement with max-min falls to 0.53 where edge memberships
    spread widely
-   Six public fuzzy graphs, one script, 114 reported values re-derived
    automatically""",
"""-   The Corollary 1 gap is a normalisation quantity, not the cost of
    binarising; the cost is measured instead
-   On planted partitions the cost of binarising tracks the fuzzy mass
    the cut discards, not the spread
-   A resolution threshold for general bipartitions, with a null-model
    test and a "no structure" outcome
-   Best paths under Archimedean t-norms are shortest paths; only
    max-min is ordinally invariant
-   Six graphs from four public corpora, planted benchmarks, one script
    that re-derives the reported values""")

# --------------------------------------------------------------- abstract
old_abs = s[s.index("## Abstract"):s.index("**Keywords:**")]
new_abs = """## Abstract

A knowledge graph built from learner interaction logs carries graded
prerequisite edges and graded, uncertain learner states, and the
pipelines that consume it discard the grades at three points: edges are
thresholded before a crisp community detector is run, the (mean,
dispersion) state a stochastic estimator returns is collapsed to one
type-1 triple, and max-min path composition is replaced by a
non-idempotent t-norm. This paper asks, for each shortcut, whether the
loss is bounded, computable from the graph, and large enough to matter,
and it answers with measurements rather than with proxies wherever the
two disagree. For binarisation, the fuzzy-graph modularity exceeds the
α-average of its level-set modularities under global normalisation by γ
times a sum of covariances of the community strength profiles
(Proposition 2); we show by a counterexample that this gap is not the
cost of binarising, since it is positive on graphs that are crisp up to
scale, and we measure the cost instead on planted partitions with known
communities and on six real graphs, refitting the resolution per level
set: the loss is governed by the fuzzy mass the cut discards, is
negligible while more than about 60% of the mass survives, and does not
depend on the membership spread. A resolution threshold for general
bipartitions is stated, with the balanced-cut form 2c/W as a special
case, and Algorithm 1 is extended by a degree-preserving null-model
test that returns "no structure detected" on graphs where every
partition is degenerate. For path composition, best paths under every
continuous Archimedean t-norm are shortest-path problems through the
additive generator, so no family is computationally privileged; what
distinguishes max-min is ordinal invariance, which we prove and verify,
and on real graphs rank agreement with max-min runs from 0.53 where the
membership scale is wide to above 0.93 where it is concentrated. The
cautious type-1 collapse of an interval-valued state is shown to be a
defect of that construction rather than of type-1 representations, and
the section is reduced to a methodological remark pending an estimator
study. One released script rebuilds every graph from public data and
re-derives the reported values.

"""
s = s.replace(old_abs, new_abs)
rep("**Keywords:** Fuzzy graphs; Community detection; Modularity; Interval\ntype-2 fuzzy sets; Triangular norms; Learner modelling",
    "**Keywords:** Fuzzy graphs; Community detection; Modularity; Resolution\nlimit; Triangular norms; Learner modelling")

# ----------------------------------------------------------- introduction
rep("""In each of these three operations the common engineering practice, which
our own earlier pipelines followed, discards the grades at the first
opportunity.""",
"""In each of these three operations a cheap practice, which our own
earlier pipelines followed, discards the grades at the first
opportunity.""")
rep("""replaced by a strict t-norm so that long paths are penalised. Each
shortcut is one line of code, runs faster than the alternative, and
produces output that looks reasonable. None of them comes with a
statement of what it costs.""",
"""replaced by a non-idempotent t-norm so that long paths are penalised.
Each shortcut is one line of code and produces output that looks
reasonable. None of them comes with a statement of what it costs.""")
rep("""alone, so that a practitioner can report it next to the result? And is
it large enough on real graphs to change a decision? A shortcut whose
cost is provably small can be kept; one whose cost is large and
measurable should be replaced, and the replacement priced.""",
"""alone, so that a practitioner can report it next to the result? And is
it large enough to matter, measured against a known answer rather than
against another proxy? A shortcut whose cost is small can be kept; one
whose cost is large and measurable should be replaced, and the
replacement priced. We were able to answer the third part with ground
truth for the first shortcut, on planted partitions; for the other two
the evidence remains proxy-based, and Section 10.2 says so.""")
rep("""modularities of those level sets. Proposition 2 shows that Q_F dominates
their α-average under global normalisation, with the gap given exactly
as γ times a sum of covariances of the community strength profiles
(Corollary 1); the gap is zero if and only if the profiles are
uncorrelated, in particular whenever μ_E is crisp, and it is computable
from the graph and the partition. The same bookkeeping yields a
closed-form lower bound on the resolution at which any non-trivial
structure can appear (Proposition 1), which explains why a resolution
tuned on a sparse curated graph returns the trivial partition on a dense
constructed one, and which Algorithm 1 turns into a selection procedure.""",
"""modularities of those level sets. Proposition 2 shows that Q_F dominates
their α-average under global normalisation, with the gap given exactly
as γ times a sum of covariances of the community strength profiles
(Corollary 1). An earlier version of this paper read that gap as the
cost of binarising. It is not: Remark 3 gives a graph on which every
membership equals c, binarising is lossless, and the gap is γ(1/c −
1)Σ_κ a_κ b_κ. The gap measures how far the memberships sit below one
under the global normalisation, so the question of what binarising
costs has to be answered empirically, and Section 9.1 does so on
planted partitions with known communities and on the real graphs, with
the resolution refitted for every level set. The same bookkeeping yields
a threshold on the resolution below which a given bipartition cannot
beat the trivial partition (Proposition 1); its balanced-cut form 2c/W
is the familiar resolution-limit quantity, it explains why a resolution
tuned on a sparse curated graph returns the trivial partition on a dense
constructed one, and Algorithm 1 turns it into a selection procedure
that now ends with a null-model test rather than with the first stable
non-trivial partition.""")
rep("""The second shortcut concerns the learner state. Converting an interval
\\[p̄ − ησ, p̄ + ησ\\] into memberships over *unknown*, *partial* and
*mastered* is a fuzzification problem, and the interval induces an
interval type-2 membership over the three labels whose footprint of
uncertainty Section 6 obtains in closed form. The common collapse is
shown to fail in two independent ways: its renormalisation step leaves
the admissible embedded region, and its cautious endpoint selection
degenerates to rejecting every candidate once the half-width exceeds a
threshold that admits a closed form in the cut-points and the gate
level. Both failures are measured on a dense grid.""",
"""The second shortcut concerns the learner state. Converting an interval
\\[p̄ − ησ, p̄ + ησ\\] into memberships over *unknown*, *partial* and
*mastered* is a fuzzification problem, and the interval induces an
interval type-2 membership over the three labels whose footprint of
uncertainty Section 6 obtains in closed form. The cautious collapse our
earlier pipelines used evaluates each shoulder label at the interval
endpoint most favourable to it; Section 6.3 shows that this rejects
every candidate once the half-width exceeds a closed-form threshold,
and that the defect belongs to that construction, not to type-1
representations: a type-1 gate on the interval image, or a probability
gate on the predictive distribution, does not share it. Section 6 is
therefore kept short, and the comparison of gates on real estimator
output is left as the outcome study of Section 10.2.""")
rep("""The third shortcut has the shortest answer. Since T(a,b) ≤ min(a,b) for
every t-norm T \\[8\\], no member of the class can let a strong edge
compensate a weak one, so the design freedom a parametric family offers
is only how far below the weakest link to erode. What a parametric sweep
can measure, and what has not to our knowledge been measured on real
fuzzy graphs, is how much that erosion reorders the induced ranking;
Section 9.3 reports that it depends almost entirely on the spread of the
membership distribution.""",
"""The third shortcut has two answers, one of which corrects our earlier
version. Since T(a,b) ≤ min(a,b) for every t-norm T \\[8\\], no member of
the class can let a strong edge compensate a weak one, so the design
freedom a parametric family offers is only how far below the weakest
link to erode. The computational objection we previously raised against
non-idempotent t-norms was wrong: every continuous Archimedean t-norm
has an additive generator, the composition along a path is the
generator image of a sum of edge lengths, and the best path is an
ordinary shortest path (Section 7.3). What does distinguish max-min is
ordinal invariance: its best-path ordering is unchanged by every
strictly increasing rescaling of μ_E, and no Archimedean t-norm shares
this (Proposition 3). Since μ_E is an evidence-pooling degree rather
than a measurement on a cardinal scale (Remark 1), that invariance is
the argument for max-min, and Section 9.3 measures how much the
alternatives reorder best paths on real graphs.""")
rep("""rebuilds every graph, reproduces every table and compares 114 printed
values against its own output.""",
"""rebuilds every graph, reproduces every table and compares the printed
values listed in the supplement against its own output.""")
# contributions
old_c = s[s.index("**Contributions.**"):s.index("Section 2 positions the work.")]
new_c = """**Contributions.**

1.  *Binarising before partitioning.* The fuzzy-graph modularity Q_F
    exceeds the α-average of the level-set modularities under global
    normalisation by γ Σ_κ Cov(f_κ, g_κ) (Proposition 2, Corollary 1); a
    counterexample shows that this gap is not the cost of binarising
    (Remark 3). The cost is measured instead, on directed planted
    partitions whose membership mean, spread and community correlation
    are varied independently, and on six real graphs, with the
    resolution refitted per level set: it is governed by the fuzzy mass
    the cut discards, and the spread of the memberships has no
    detectable effect (Tables 6--7).

2.  *Resolution.* A bipartition beats the trivial partition exactly when
    γ exceeds a threshold θ(B) that reduces to 2c/W for balanced cuts
    (Proposition 1); the threshold is the split gain of generalised
    modularity, stated here for fuzzy mass. Algorithm 1 uses it to seed
    the resolution and now ends with a degree-preserving null-model
    test, returning "no structure detected" on graphs where every
    partition is degenerate rather than a near-singleton partition.

3.  *Composition.* Best paths under every continuous Archimedean t-norm
    are shortest-path problems through the additive generator, at the
    cost of the max-min sweep (Section 7.3); the earlier claim of an
    exponential cost is withdrawn. Max-min is the only operator among
    those considered whose best-path ordering is invariant under
    strictly increasing rescalings of μ_E (Proposition 3), and on real
    graphs the alternatives reorder best paths from negligibly (rank
    agreement above 0.93) to substantially (0.53).

4.  *Verification and negative results.* All claims are checked on six
    real fuzzy graphs from four public corpora (up to 33575 edges on 867
    nodes), on planted partitions and on synthetic graphs to 20000
    nodes; the resolution defaults do not transfer between densities;
    the prerequisite reading of μ_E holds at the item level but not at
    the skill level of the ASSISTments family; and the cautious type-1
    collapse of an uncertain state degenerates where simpler gates do
    not.

"""
s = s.replace(old_c, new_c)
rep("""Sections 5, 6 and 7 treat the three shortcuts in turn. Section 8 states
the construction pipeline, Section 9 the experiments, and Sections 10
and 11 discuss and conclude.""",
"""Sections 5, 6 and 7 treat the three shortcuts in turn, Section 6 in the
form of a methodological remark. Section 8 states the construction
pipeline, Section 9 the experiments, and Sections 10 and 11 discuss and
conclude.""")

# ----------------------------------------------------------- related work
rep("""Modularity at a fixed resolution has a known resolution limit \\[19\\],
and the interaction of resolution with graph density is well documented
\\[20\\]. What Proposition 1 adds is a closed form for the smallest
resolution at which a bipartition can beat the trivial partition on a
fuzzy graph, together with a procedure that uses it; we have not found
this bound stated for weighted or fuzzy graphs.""",
"""Modularity with non-negative weights is Newman's weighted modularity
\\[44\\], and Q_F is that functional with memberships as weights.
Modularity at a fixed resolution has a known resolution limit \\[19\\];
the resolution parameter of Reichardt and Bornholdt \\[45\\] and the
resolution profiles of Arenas et al. \\[46\\] address it by scanning,
Traag et al. \\[47\\] by a resolution-limit-free objective, and
consensus clustering \\[48\\] by aggregating many runs. The split gain
behind Proposition 1 is the local-moving gain of generalised modularity
that Louvain and Leiden evaluate; what Section 5.3 does is state it for
a bipartition of a fuzzy graph, read off the threshold, and use it to
seed a resolution. We make no claim of novelty for the identity. The
interaction of resolution with density is documented in \\[20\\].""")
rep("""The contribution
of Section 7 and Section 9.3 is not the bound but its measured
consequence: how much the choice within the class reorders best paths on
real fuzzy graphs, and how that depends on the membership spread.""",
"""Best paths under max-min and under
generator-based compositions are instances of the algebraic path
problem over a dioid \\[49\\], which is why both reduce to
Dijkstra-type sweeps (Section 7.3). The contribution of Section 7 and
Section 9.3 is the ordinal-invariance argument for max-min
(Proposition 3) and the measured reordering the alternatives cause on
real fuzzy graphs.""")

# ---------------------------------------------------------- Section 5.3
old53 = s[s.index("### 5.3 The smallest useful resolution"):s.index("### 5.4 Relation to the α-cuts")]
new53 = """### 5.3 The resolution threshold of a bipartition

**Proposition 1 (resolution threshold of a bipartition).** *Let B = {κ,
κ̄} be a bipartition of C with inter-community fuzzy mass c(B) (both
directions counted), and write a_1 = S_κ\\^{out}/W, b_1 = S_κ\\^{in}/W
and a_2, b_2 likewise for κ̄. Then Q_F(B) exceeds the value 1 − γ of the
trivial partition if and only if*

$$\\gamma\\mspace{6mu} > \\mspace{6mu}\\theta(B)\\mspace{6mu} = \\mspace{6mu}\\frac{c(B)/W}{a_{1}b_{2} + a_{2}b_{1}}.$$

*For a strength-balanced split (a_i = b_i = 1/2) the threshold is θ(B)
= 2c(B)/W. Consequently no bipartition improves on the trivial partition
below γ = min_B θ(B), and 2c_bal/W, with c_bal the minimum balanced
cut, is an upper bound on that onset, not a lower bound: an unbalanced
block that is weakly attached to the rest can have θ(B) far below the
balanced value.*

*Proof.* For the trivial partition, (8) gives Q_F = 1 − γ. For B, Q_F =
(W − c)/W − γ (a_1 b_1 + a_2 b_2), and since (a_1 + a_2)(b_1 + b_2) = 1
the improvement over the trivial partition is

$$\\Delta Q_{F}\\mspace{6mu} = \\mspace{6mu} - \\,\\frac{c}{W}\\mspace{6mu} + \\mspace{6mu}\\gamma\\,\\left( a_{1}b_{2} + a_{2}b_{1} \\right),$$

which is positive exactly under the stated condition. The denominator is
maximised, at 1/2, by a_1 = b_1 = 1/2, so among splits of equal cut
mass the balanced one has the smallest threshold; a split with smaller
cut mass can nevertheless have a smaller threshold. ∎

ΔQ_F is the split gain that Louvain and Leiden evaluate in their
local-moving step \\[6,7\\], written for a single bipartition of a fuzzy
graph; the resolution limit \\[19,45\\] is its consequence for
communities of fixed size. The proposition is recorded because it
settles one practical question in closed form: the balanced-cut
fraction of a graph tells a practitioner the order of magnitude of γ
below which nothing but the trivial partition can be returned. On a
sparse modular graph the best balanced bipartitions cut a small fraction
of the mass and γ well below 1 suffices; on a dense graph in which every
balanced bipartition carries close to half the mass, 2c/W approaches 1
and only γ of order 1 or larger can return anything but the trivial
partition. Table 4 shows the caveat as well as the rule: on the 2017
item-level graph Fuzzy Leiden returns 16 communities with Q_F = 0.723 at
γ = 0.5, below the balanced threshold 2c/W = 0.531 of Table 5, because
the partition it finds is not a balanced bipartition and its own θ is
smaller. The released code reports θ(B) for the spectral bisection and
for every "one community against the rest" split of the accepted
partition, and the minimum over those candidates is the number to
compare with γ.

The threshold converts into a seeding procedure: (1) compute one cheap
strength-balanced bipartition of the fuzzy graph, a single spectral
bisection of the symmetrised adjacency, and its cut fraction c/W; (2)
set the coarse resolution γ₁ to a small multiple of 2c/W and the fine
resolution γ₂ to a fixed multiple of γ₁; (3) confirm non-degeneracy by
checking that the returned partition is non-trivial and stable over a
handful of seeds, moving one step up a logarithmic grid if it is not.
Steps (1) to (3) are the earlier version of Algorithm 1, and they
cannot fail: step (3) rejects the trivial partition, and at a large
enough γ Leiden returns a near-singleton partition deterministically,
which is non-trivial and stable. On a graph without community structure
the procedure therefore replaces one degenerate output by another,
which is what Table 5 shows on the three KC-level graphs (74 to 87
communities on 83 to 110 nodes, negative Q_F). The revised Algorithm 1
adds a fourth step: the accepted partition is tested against a
degree-preserving null model that keeps the in- and out-degree
sequence and the empirical membership distribution and reassigns the
memberships to edges at random, and the procedure returns "no structure
detected" when the observed Q_F is not separated from the null
distribution.

    Algorithm 1  Resolution from the cut threshold, with a null-model test
    Input:  fuzzy graph G_F; multipliers ρ1 ≥ 1, ρ2 > 1 (defaults 1.25, 2);
            seeds S; grid factor g = 2; stability threshold θ_ARI = 0.8;
            null draws N (default 20); separation z_min (default 3)
    1:  B ← strength-balanced spectral bisection of the symmetrised adjacency of G_F
    2:  c/W ← inter-block fuzzy mass of B divided by the total fuzzy mass W
    3:  γ1 ← ρ1 · (2c/W);   γ2 ← ρ2 · γ1
    4:  repeat
    5:      run Fuzzy Leiden at γ1 once per seed in S
    6:      if every run returns a non-trivial partition and the pairwise
               adjusted Rand index across seeds is at least θ_ARI then break
    7:      γ1 ← g · γ1;   γ2 ← g · γ2
    8:  Q_obs ← best Q_F over S at γ1
    9:  for i = 1..N: rewire G_F preserving degrees, permute the memberships
            over the edges, run Fuzzy Leiden at γ1 over S, record the best Q_F
    10: z ← (Q_obs − mean of the N null values) / their standard deviation
    11: if z < z_min: return "no structure detected"
    Output: resolutions (γ1, γ2) and the partition at γ1, or "no structure detected"

The loop of steps 4 to 7 is capped at twelve doublings in the released
implementation; no real graph in Section 9 reached the cap. Symmetrising
the adjacency in step 1 does not bias the estimate on directed graphs,
since c/W is a mass quantity and orientation-free. Section 9.1 reports
the revised algorithm on planted graphs with and without structure, on
a dense co-practice-like graph, and on the six real graphs.

"""
s = s.replace(old53, new53)

# ---------------------------------------------------------- Section 5.4
rep("""The corollary turns the qualitative bound into a computable one: the
cost of binarising a fuzzy graph before partitioning it is at most γ
times the summed dispersion of the α-profiles of the community
strengths, quantities available from the graph and the partition alone,
so the bound can be reported alongside Q_F. Crisp memberships give
constant profiles, zero variance and zero gap, recovering the equality
case; a graph whose memberships spread widely over (0,1\\] admits a large
gap, which is exactly the regime in which thresholding is most
destructive.

Two readings are worth separating. Analytically, Proposition 2 says that
fuzzy-graph modularity is not the α-average of crisp modularities but an
upper bound on it, and that the gap is exactly the correlation gap of
Chebyshev's inequality applied to the null-model term, reducing to a
Jensen gap in the symmetric case: thresholding a fuzzy graph and
averaging the resulting crisp modularities systematically understates
the structure the fuzzy graph carries, because the null model is
bilinear in the strength profiles and thresholding disperses them.
Practically, it means that the common expedient of binarising a fuzzy
graph at some α and running a crisp community detector is not a neutral
simplification of Equation (7), and that the direction of the bias is
known.""",
"""The corollary makes the gap computable from the graph and the partition.
What the gap means is a separate question, and the answer corrects the
earlier version of this paper, which read it as the cost of binarising.

**Remark 3 (the gap is not the cost of binarising).** Let every edge
carry the same membership c ∈ (0,1). Q_F is invariant to rescaling μ_E
by a constant, so Q_F(Π) equals the crisp modularity of the support
graph for every Π, and thresholding at any α ≤ c returns that support
graph: binarising is lossless. Yet f_κ(α) = (a_κ/c)·1\\[α ≤ c\\] with a_κ
the out-strength share of κ, and likewise g_κ, so Cov(f_κ, g_κ) =
a_κ b_κ (1/c − 1) and the gap equals γ(1/c − 1)Σ_κ a_κ b_κ, which is
0.25γ for c = 0.5 and two balanced communities. The released code
confirms the identity numerically to machine precision on a planted
graph (Section 9.1). The gap therefore vanishes for μ_E ∈ {0,1} but
not for a graph that is crisp up to scale; it measures how far the
memberships lie below one under the global normalisation of (9), and
it is neither a function of the membership spread nor a loss incurred
by any crisp detector. The phrase "zero only in the crisp case" in the
earlier version was wrong in both directions: the equality condition of
Proposition 2 is that for each community one profile is almost
everywhere constant, which crisp memberships satisfy but which is not
necessary, and a constant-membership graph is lossless to binarise yet
has a positive gap.

What Proposition 2 does say is representation-theoretic: fuzzy-graph
modularity is not the α-average of its level-set surrogates but an
upper bound on it, the gap being the correlation gap of Chebyshev's
inequality applied to the null-model term, reducing to a Jensen gap in
the symmetric case. The "tightness" of the Cauchy--Schwarz bound
reported in Section 9.1 is likewise expected rather than informative:
on the near-reciprocal ASSISTments graphs f_κ ≈ g_κ, so Cov ≈ Var and
the bound is an identity up to the asymmetry. Whether binarising costs
anything is an empirical question about the partition a crisp detector
returns, and Section 9.1 answers it with ground truth.""")
rep("""The normalisation in (9) is by the fuzzy mass W rather than by \\|E_α\\|,
and that choice is forced rather than convenient.""",
"""The normalisation in (9) is by the fuzzy mass W rather than by \\|E_α\\|,
and that choice is forced by the proof rather than by convenience. It
also means that (9) is not the modularity of the crisp graph G_α:
writing r = \\|E_α\\|/W, the internal term of (9) is the crisp one
weighted by r and the null term by r², so Q̃(α; Π) tends to zero as α →
1 simply because edges disappear, whereas the crisp modularity of G_α
need not.""")
rep("""The mass weighting also has the right reading:
Table 6 in Section 9.1 shows an α = 0.6 cut of the 2009 item-level graph
retaining 52 of 4950 edges, a slice whose per-cut modularity is high
precisely because almost nothing of the graph remains in it.

Lemma 1 compares two evaluations of the *same* partition. The shortcut a
practitioner takes is to run a crisp detector on one level set and keep
whatever partition it returns, which is a different partition; whether
it is a worse one for the fuzzy graph is an empirical question, and
Table 6 answers it on the six real graphs by evaluating the level-set
partition under Q_F on the fuzzy graph it was meant to partition.""",
"""Proposition 2 compares two evaluations of the *same* partition. The
shortcut a practitioner takes is to run a crisp detector on one level
set, with a resolution chosen for that level set, and keep whatever
partition it returns; whether that partition is worse for the fuzzy
graph, and whether it recovers the planted communities less well, are
empirical questions. Section 9.1 defines the cost operationally as

$$\\text{loss}(\\alpha)\\mspace{6mu} = \\mspace{6mu} Q_{F}\\left( \\Pi_{\\text{fuzzy}} \\right) - \\max_{\\gamma\\prime}Q_{F}\\left( \\Pi_{\\alpha}(\\gamma\\prime) \\right),$$

the resolution γ′ being refitted for each level set, together with the
agreement between Π_α and Π_fuzzy and, where a planted partition
exists, the recovery of the truth by each.""")

# ---------------------------------------------------------- Section 6.3
rep("""### 6.3 Two defects of the type-1 collapse

Equations (11)--(12) select, for each extreme label, the endpoint at
which that label's membership is largest, and give μ_par the remainder.
This is a type reduction by a fixed worst-case-toward-caution criterion
rather than by Karnik--Mendel iteration \\[22\\], which is why it costs
one comparison instead of an inner loop. Whether the result is an
admissible embedded type-1 set of the footprint it comes from is a
separate question, and the answer is no, for two independent reasons.""",
"""### 6.3 The cautious collapse, and why its defect is its own

Equations (11)--(12) are the construction our earlier pipelines used;
we know of no published pipeline that uses exactly this collapse, and
the earlier version of this paper was wrong to call it common practice.
They select, for each shoulder label, the endpoint at which that
label's membership is largest, and give μ_par the remainder. This is a
type reduction by a fixed worst-case-toward-caution criterion rather
than by Karnik--Mendel iteration \\[22\\]. Whether the result is an
admissible embedded type-1 set of the footprint it comes from is a
separate question, and the answer is no, for two reasons; the second is
the one that matters for a decision.""")
rep("""Section 9.5 measures the failure rate at 7.3% of the state space. Note
that no amount of property-based testing of the collapsed triple could
have found this: sum-to-one, boundedness and monotonicity are all
properties internal to the triple.""",
"""Section 9.5 measures the failure rate on a grid. Embeddedness is a
criterion internal to the type-2 representation, and we make no claim
that violating it harms a decision; it is recorded because it is a
cheap diagnostic.""")
rep("""For the default cut-points and τ_ZPD = 0.4 this gives h\\^\\* = 0.075,
which the measured grid brackets: Table 12 shows 16.9% of candidates
still admitted at ησ = 0.05 and none at ησ = 0.10 or beyond. The
mechanism is therefore general --- for any trapezoidal partition and any
τ_ZPD \\< 1 there is a finite epistemic-uncertainty width beyond which
the type-1 collapse recommends nothing to anyone --- while the
particular numbers 0.075 and 7.3% are instances at the default
cut-points. The construction inverts the intended reading of "defensive"
precisely for the learners whose state is least well determined, and the
inversion is invisible while the state is represented as a single
triple.

On this evidence we adopt the IT2 representation, retaining the type-1
collapse as the ησ = 0 special case, where the two coincide. The cost is
measured rather than assumed: the state grows from three scalars per KC
to six, and constructing the footprint costs 2.79× the collapse, or
4.23× if a Nie--Tan reduction \\[23\\] is applied to recover a defuzzified
band for reporting (Section 9.5). Nie--Tan is chosen over Karnik--Mendel
type reduction \\[22,24\\] on cost grounds, a single pass over the bounds
against an iteration to the exact centroid; on this construction the
Nie--Tan point differs from the midpoint of the Karnik--Mendel centroid
interval by at most 0.029 over half-widths up to 0.15 and by at most
0.015 over the operating range ησ ≤ 0.10, an order of magnitude below
the coarsest decision margin in Table 12. Type reduction plays no role
in the gate itself: the three-valued outcome of Section 6.2 is computed
from the bounds (μ̲\\^{par}, μ̄\\^{par}) directly, so no reduction can
collapse the indeterminate class, and the final column of Table 12
measures how often a decision taken on the Nie--Tan point would agree
with the collapse. The max-min composition of Section 7 and the path
score of Section 8.3 extend to interval arithmetic without modification,
min and max being monotone in each argument and acting on the bounds
separately.""",
"""For the default cut-points and τ_ZPD = 0.4 this gives h\\^\\* = 0.075,
which the measured grid brackets (Table 12). The mechanism is a
three-line consequence of the construction: widening the interval can
only raise the two shoulder memberships, hence only lower μ_par. It is
therefore a defect of the cautious collapse and not of type-1
representations in general, and the comparison that matters is with
gates that do not share the design. Section 9.5 compares four: the gate
on the point estimate p̄ alone; the type-1 gate on the interval image,
pass iff min_{x ∈ \\[lo, hi\\]} m_par(x) ≥ τ_ZPD, which is the "pass"
rule of the interval type-2 gate and needs no type-2 machinery; the
probability that the state lies in the ZPD band under a normal
predictive distribution with the dropout moments, thresholded at one
half; and the expected membership over the interval. On the grid the
cautious collapse and the interval-image gate admit exactly the same
states, so what the interval type-2 form adds over the simplest type-1
repair is not a larger pass set but the three-valued split of the
remainder into *fail* and *indeterminate*; the probability gate keeps
admitting candidates to ησ = 0.20 and the point-estimate gate at all
widths, since neither is designed to be cautious. Which of these is
right depends on whether σ is a calibrated dispersion and on what an
abstention is worth downstream, and neither can be settled on a grid.
Two further corrections are in order. The widening factor η = 1
selects the core of π_C by Lemma 2(iii), but the Chebyshev guarantee at
that level is P(\\|X − p̄\\| \\< ησ) ≥ 1 − η\\^{−2} = 0, so the choice has
no coverage content; η = 2 and η = 3 give 0.75 and 0.89, and Table 1
now lists η as a label-semantics choice. And with R = 8 dropout passes
the relative standard error of σ is about 27%, while dropout dispersions
are known to be poorly calibrated under shift \\[50\\], so Lemma 2's
semantics presuppose a σ that Section 9 never checks against held-out
outcomes. For these reasons the interval type-2 state is kept in the
released code as a representation and its gate comparison on a trained,
evaluated estimator is deferred to the outcome study of Section 10.2;
the footprint and Nie--Tan reduction \\[23\\] cost a few hundredths of a
millisecond per state and are not a consideration.""")

# ---------------------------------------------------------- Section 7
rep("""### 7.2 Containment of the operators in use""", """### 7.2 Containment of the operators in use""")
rep("""A
comparison restricted to the three standard operators therefore spans
both families at their endpoints; what a sweep adds is the interior, and
Section 9.3 reports it on real graphs.""",
"""A
comparison restricted to the three standard operators therefore spans
both families at their endpoints; what a sweep adds is the interior, and
Section 9.3 reports it on real graphs. Terminology: the product and the
Schweizer--Sklar norms with p \\< 0 are strict (continuous and strictly
increasing on (0,1\\]²); Łukasiewicz, the Yager family and
Schweizer--Sklar with p \\> 0 are nilpotent. All are continuous
Archimedean, and "non-idempotent" is the common property that separates
every one of them from the minimum.

### 7.3 Best paths under Archimedean t-norms are shortest paths; only max-min is ordinally invariant

Every continuous Archimedean t-norm T has an additive generator g :
\\[0,1\\] → \\[0, ∞\\], continuous and strictly decreasing with g(1) = 0,
such that T(a, b) = g\\^{(−1)}(g(a) + g(b)), where g\\^{(−1)}(y) =
g\\^{−1}(min(y, g(0))) is the pseudo-inverse \\[8\\]. By associativity
the composition along a path π is S_T(π) = g\\^{(−1)}(Σ\\_{e ∈ π}
g(μ_E(e))), and since g\\^{(−1)} is non-increasing the best path from a
to b under T is a path of minimum additive length Σ g(μ_E(e)) with
non-negative edge lengths: an ordinary shortest-path problem, solved
exactly by Dijkstra's algorithm in O(\\|E\\| log \\|C\\|), and with a hop
limit L by L rounds of Bellman--Ford relaxation, which is also exact
because non-negative lengths mean that the minimum over walks of at most
L hops is attained by a simple path. Table 13 lists the generators of
the families used in this paper. The earlier version of this paper
stated that only the minimum has optimal substructure and that exact
best paths under the other t-norms require enumeration at an
exponential cost; that statement was wrong, and the 350-fold figure it
reported measured the cost of our enumeration, not of the t-norms.
Section 9.3 reports the corrected cost.

**Table 13.** Additive generators of the t-norm families used; the
best path under T minimises Σ g(μ_E) along the path.

  -------------------------------------------------------------------------
  Family                  g(x)                 g(0)        Kind
  ----------------------- -------------------- ----------- ----------------
  product                 −ln x                ∞           strict

  Łukasiewicz             1 − x                1           nilpotent

  Yager, p \\> 0           (1 − x)\\^p            1           nilpotent

  Schweizer--Sklar, p \\< 0 (1 − x\\^p)/p          ∞           strict

  Schweizer--Sklar, p \\> 0 (1 − x\\^p)/p          1/p         nilpotent

  minimum                 ---                  ---         idempotent, not
                                                           Archimedean
  -------------------------------------------------------------------------

What does separate max-min from the rest is invariance under rescaling
of the membership scale. Remark 1 reads μ_E as an evidence-pooling
degree: the OWA aggregation of three signals fixes its order but not
its unit, and Section 9.4 shows that the orness of the aggregation
moves the mean membership from 0.34 to 0.57 without changing the
retained edge set. A composition operator that respects only the order
of μ_E gives the same best paths under every such change; one that uses
the differences does not.

**Proposition 3 (ordinal invariance).** *Let φ : \\[0,1\\] → \\[0,1\\] be
strictly increasing and let G_F\\^φ carry memberships φ ∘ μ_E. (i) For
every pair (a, b) the sets of max-min best paths on G_F and G_F\\^φ
coincide, and the ordering of targets by μ_E\\^∞(a, ·) is the same on
both graphs. (ii) For every continuous Archimedean t-norm T there exist
a graph and a strictly increasing φ for which the T-best path from a to
b changes.*

*Proof.* (i) S_min(π) = min_{e ∈ π} μ_E(e), and φ(min_e μ_E(e)) = min_e
φ(μ_E(e)) because φ is increasing, so S_min\\^φ(π) = φ(S_min(π)) for
every path, and φ preserves the order of path strengths and of target
strengths. (ii) Take a two-hop path π_1 with memberships (x, x) and a
one-hop path π_2 with membership y, x \\> y. Under T, π_1 beats π_2 iff
2g(x) \\< g(y). Since g is continuous with g(1) = 0 and g(0) \\> 0,
choose x close to 1 so that 2g(x) \\< g(y); then choose φ increasing
with φ(x) = x′ and φ(y) = y′ such that 2g(x′) \\> g(y′), which is
possible because g(x′) → g(y′) as x′ → y′ from above while g(y′) stays
positive; the T-best path changes from π_1 to π_2. ∎

Section 9.3 verifies both parts on a synthetic graph under random
increasing piecewise-linear rescalings: the max-min ordering of targets
is unchanged (Kendall τ = 1 on every source and every map) and every
Archimedean operator reorders. The proposition is the argument for
max-min when μ_E is ordinal, and it also says when a non-idempotent
t-norm is legitimate: when μ_E carries a cardinal, probability-like
scale, so that differences and products of memberships mean something.
The construction of Section 8.1 does not provide such a scale, and the
sweep of Section 9.3 should be read with that in mind.""")

# ------------------------------------------------------------ Section 8.4
rep("""Where expert labels are available they are fitted by
maximising Cohen's κ between the induced band and the expert-assigned
level. \\[CẦN XÁC MINH: τ_E, R, τ_path ghi nguồn "D"; cần nêu tập
validation và tiêu chí đã dùng để chọn, hoặc đổi nguồn thành "P" nếu
thực tế chọn theo kinh nghiệm.\\]""",
"""Where expert labels are available they are fitted by
maximising Cohen's κ between the induced band and the expert-assigned
level. The pruning threshold τ_E, the number of dropout passes R and the
path-edge floor τ_path were set by hand in earlier work and carried
over; no validation split was used to choose them, and Table 1 now
records them as practitioner defaults (P) rather than as fitted
values. The orness sweep of Section 9.4 shows that τ_E is insensitive
to the aggregation; the sensitivity of the results to R and τ_path has
not been measured.""")
sub(r"  τ_E            Edge-prune     0\.2            §8\.1           D\n",
    "  τ_E            Edge-prune     0.2            §8.1           P\n")
sub(r"  R              MC-dropout     8              §8\.2           D \(variance\n                 passes                                       plateau\)\n",
    "  R              MC-dropout     8              §8.2           P\n                 passes                                       \n")
sub(r"  η              Uncertainty    1\.0            Eq\. \(10\)       T \(core of\n                 widening                                     π_C, Lemma 2\)\n",
    "  η              Uncertainty    1.0            Eq. (10)       P (core of\n                 widening                                     π_C; no\n                                                              coverage)\n")
sub(r"  τ_path         Minimum path   0\.3            §8\.3           D\n",
    "  τ_path         Minimum path   0.3            §8.3           P\n")

# ---------------------------------------------------------- Section 9 intro
rep("""Tables 2, 3, 4, 5, 6, 8, 9 and 10 are computed from real public data.
Table 7 is computed from synthetic graphs and Tables 11 and 12 from
dense parameter grids generated by the released code; all three are
labelled as such.""",
"""Tables 2, 3, 4, 5, 6, 9, 10 and 11 are computed from real public data;
the six real graphs derive from four corpora (two granularities each of
ASSISTments 2009 and 2017, one 2015 graph, one MOOCCubeX graph). Tables
7 and 8 are computed from planted and synthetic graphs, and Tables 12
and 14 from dense parameter grids generated by the released code; all
are labelled as such.""")

# ---------------------------------------------------------- Section 9.1
rep("""Every one of
the 10 communities induces a weakly connected subgraph of the support
graph, with zero counterexamples, which is the computational counterpart
of Lemma 1(i).""",
"""Every one of
the 10 communities induces a weakly connected subgraph of the support
graph, which is the sanity check for Lemma 1(i).""")
rep(""""Mono." is the Proposition
2(iii) check over ten successive optimisation iterations at every
resolution.""", """"Mono." is the Lemma 1(iv)
check over ten successive optimisation iterations at every resolution.""")
rep("""*Both propositions hold on real data at scale.* Across five graphs and
four resolutions, twenty partitions in total, every returned community
induces a weakly connected subgraph of the support graph, with zero
counterexamples, on graphs up to 867 nodes and 33575 fuzzy edges. Over
ten successive optimisation iterations per configuration, Q_F never
decreased. Note that boundedness is visible in the table as well: the
negative entries at γ = 5.0 lie above −γ, consistent with Proposition
2(i).""",
"""*The inherited properties were not violated.* Across five graphs and
four resolutions, twenty partitions in total, every returned community
induces a weakly connected subgraph of the support graph, on graphs up
to 867 nodes and 33575 fuzzy edges, and over ten successive
optimisation iterations per configuration Q_F never decreased. These
are sanity checks on the implementation of a proved lemma, not
evidence for it; boundedness is visible in the table as well, the
negative entries at γ = 5.0 lying above −γ as Lemma 1(ii) requires.""")
rep("""The cause is
density, and Proposition 1 gives it quantitative form, now with the
thresholds measured: the balanced-cut fractions Algorithm 1 measures on
the three KC-level graphs are c/W = 0.346, 0.342 and 0.366, so the
thresholds 2c/W are 0.691, 0.684 and 0.732 --- all above 0.5 and all
below 1 --- and the table behaves exactly as the bound dictates,
degenerate at γ = 0.5 and non-trivial from γ = 1.0 upward, with fine
structure from γ = 2.""",
"""The cause is
density, and Proposition 1 gives it quantitative form: the balanced-cut
fractions Algorithm 1 measures on the three KC-level graphs are c/W =
0.346, 0.342 and 0.366, so the balanced thresholds 2c/W are 0.691, 0.684
and 0.732 --- all above 0.5 and all below 1 --- and the table is
degenerate at γ = 0.5 and non-trivial from γ = 1.0 upward. The balanced
value is an upper bound on the onset (Proposition 1), so this is
consistent rather than dictated: the 2009 KC-level graph returns a
two-community split at γ = 0.5 below its balanced threshold, and the
2017 item-level graph returns 16 communities at γ = 0.5 below its
balanced threshold 0.531, in both cases through unbalanced splits with
a smaller θ.""")
rep("""The item-level graphs behave as
a prerequisite structure should, with Q_F between 0.18 and 0.94 across
the resolution grid and communities of interpretable size, while the
KC-level graphs behave like dense co-practice graphs, which is what they
are:""",
"""The item-level graphs return
partitions with Q_F between 0.18 and 0.94 across the resolution grid,
while the KC-level graphs behave like dense co-practice graphs, which
is what they are. Whether the 28 communities of the 2009 item-level
graph at every resolution are prerequisite structure or the
skill-builder problem sets the items belong to is a question we could
not settle from Q_F alone; the released code now reports the weakly
connected components and the problem-set partition of every graph next
to its partitions, and Section 10.3 lists the comparison as a
limitation until it is run against a reference partition. The
co-practice reading of the KC-level graphs is the following:""")
rep("""*Algorithm 1 removes the degenerate rows and exposes a second regime.*
Table 5 reports Algorithm 1 on all six real graphs with its default
multipliers (ρ₁ = 1.25, ρ₂ = 2, five seeds, θ = 0.8). The cut fraction
it measures separates the graphs into two groups. On the three graphs
with c/W ≤ 0.27 (MOOCCubeX-CS and the two item-level graphs) the seeded
resolution is accepted within at most one doubling and the returned
partition has Q_F between 0.47 and 0.96. On the three KC-level graphs,
c/W ≈ 0.35, the threshold 2c/W ≈ 0.7 is cleared only after two
doublings, at γ₁ ≈ 3.4--3.7, and the partition the algorithm then
accepts as non-trivial and seed-stable has 74 to 87 communities and a
*negative* Q_F. That outcome is informative rather than a failure of the
procedure: the algorithm does what it promises, which is to return a
stable non-trivial partition when one exists at some resolution, and the
sign of Q_F at that resolution says that the stable partition is finer
than the null model expects, which is what a co-practice graph with no
community structure looks like. Section 9.1's granularity finding,
stated below, is the same fact seen from the construction side.""",
"""*Algorithm 1 seeds the resolution and, with the null-model test,
distinguishes structure from its absence.* Table 5 reports the seeding
steps of Algorithm 1 on all six real graphs with the default
multipliers (ρ₁ = 1.25, ρ₂ = 2, five seeds, θ_ARI = 0.8). On the three
graphs with c/W ≤ 0.27 (MOOCCubeX-CS and the two item-level graphs) the
seeded resolution is accepted within at most one doubling and the
returned partition has Q_F between 0.47 and 0.96. On the three
KC-level graphs the loop accepts only after two doublings, at γ₁ ≈
3.4--3.7, a partition with 74 to 87 communities on 83 to 110 nodes and a
negative Q_F; the doublings are caused by the seed-stability test, not
by the threshold, since γ₁ = 1.25 · 0.69 ≈ 0.86 already clears it. The
earlier version of this paper called that outcome informative. It is
the second degenerate output Section 5.3 warns of, and the null-model
step of the revised algorithm is what names it. Table 7 reports the
revised algorithm where its behaviour can be checked: on a planted
partition with p_in = 0.8 the observed Q_F lies far outside the null
distribution and the outcome is "structure"; on the same construction
with p_in = 1/k, so that no communities are planted, the seeding loop
climbs to γ₁ ≈ 109 and a 582-community partition, and the null test
returns "no structure detected" (z = 0.6); on a dense co-practice-like
graph of 110 nodes and 50 edges per node it accepts a 109-community
partition with Q_F = −0.041 and the null test again returns "no
structure detected" (z = −0.1). The same test on the six real graphs is
run by the released script; its outcome on the KC-level graphs decides
whether the co-practice reading below stands. \\[CẦN CHẠY: `python
scripts/null_model_check.py data/graphs/*_edges.csv` trên sáu đồ thị
thật và điền z, outcome vào Bảng 7 trước khi nộp.\\]

**Table 7.** The revised Algorithm 1 on planted and synthetic graphs
(ρ₁ = 1.25, ρ₂ = 2, five seeds, 20 null draws, z_min = 3). θ_spec is
the threshold of Proposition 1 for the spectral bisection (equal to
2c/W for a balanced split) and θ_min the minimum over the "one
community against the rest" splits of the accepted partition.

  --------------------------------------------------------------------------------------------
  Graph                      γ₁ accepted  Communities  Q_F at γ₁  z       Outcome               θ_spec  θ_min
  -------------------------- ------------ ------------ ---------- ------- --------------------- ------- -------
  Planted, 600 nodes, 12     0.258        10           0.825      \\> 10³   structure             0.207   0.155
  communities, p_in = 0.8

  Same, p_in = 1/12 (no      108.9        582          −0.179     0.6     no structure detected 0.680   0.680
  planted structure)

  Dense co-practice-like,    4.54         109          −0.041     −0.1    no structure detected 0.909   0.909
  110 nodes, 50 edges/node

  ASSISTments and MOOCCubeX  Table 5      Table 5      Table 5    ---     \\[CẦN CHẠY\\]          ---     ---
  graphs
  --------------------------------------------------------------------------------------------
""")
rep("""**Table 5.** Algorithm 1 on the six real fuzzy graphs (ρ₁ = 1.25, ρ₂ =
2, five seeds, θ = 0.8).""", """**Table 5.** Seeding steps of Algorithm 1 on the six real fuzzy graphs
(ρ₁ = 1.25, ρ₂ = 2, five seeds, θ_ARI = 0.8).""")

# Table 6 paragraph and new benchmark
rep("""*Binarising before partitioning, measured.* Proposition 2 compares Q_F
with the α-average of level-set modularities *for the same partition*.
The practitioner's shortcut is different: threshold the graph at one α,
run crisp Leiden on E_α, and keep the partition that returns. Table 6
measures that shortcut directly on the six real graphs at the
Algorithm-1 resolution of Table 5. For each α ∈ {0.2, 0.4, 0.6, 0.8}
crisp Leiden is run on the level set with unit weights (best of five
seeds), and the returned partition Π_α is evaluated two ways: by the
crisp modularity Q_α of E_α that it optimised, and by Q_F on the fuzzy
graph it was meant to partition. The second number is the one that
matters, since the downstream use consumes the fuzzy graph, and its
distance from the Fuzzy Leiden value in the same row is the measured
cost of the shortcut.""",
"""*Binarising before partitioning, measured against ground truth.*
Proposition 2 compares Q_F with the α-average of level-set surrogates
*for the same partition*, and Remark 3 shows that the gap is not a
loss. The practitioner's shortcut is different: threshold the graph at
one α, run crisp Leiden on E_α with a resolution chosen for E_α, and
keep the partition that returns. Table 6 reports the shortcut on
planted partitions where the answer is known. Directed graphs of 1000
nodes in 20 equal communities, eight out-edges per node of which 70%
are internal, carry Beta-distributed memberships whose mean (0.3 to
0.9) and standard deviation (0.05 to 0.25) are set independently, and
whose community information is either absent (shift 0: internal and
external edges have the same distribution) or present (shift 0.5: the
mean of internal edges is raised and of external edges lowered by half
the available room). Fuzzy Leiden on the fuzzy graph and crisp Leiden
on each level set are each run over γ′ ∈ {0.25, 0.5, 1, 2, 4} and
three seeds, and for each the partition with the best Q_F on the fuzzy
graph is kept, so the resolution is refitted per level set; weighted
Leiden on the memberships is Fuzzy Leiden by definition of Q_F. Cuts
are taken at absolute levels α ∈ {0.4, 0.6, 0.8} and, scale-free, at
the quartiles of the membership distribution. Each cell reports the
normalised mutual information with the planted partition and the
operational loss of Section 5.4, averaged over three graph seeds; the
Corollary 1 gap of the fuzzy partition is reported alongside.

**Table 6.** The binarise-then-partition shortcut on planted
partitions, resolution refitted per level set. "Fuzzy" is Fuzzy Leiden
(NMI with the truth; Corollary 1 gap of that partition). Each cut cell
gives NMI with the truth / loss(α) / share of the fuzzy mass the cut
retains. A dash means no edge survives the cut.

""" + "@@TABLE6@@" + """

The planted benchmark settles the reading. Loss and recovery are
governed by the fuzzy mass the cut retains: Spearman's correlation
between loss and retained mass over the 420 cut cells is −0.81, between
NMI and retained mass +0.80. While a cut keeps more than about 60% of
the mass the shortcut is free (mean loss below 0.01, NMI above 0.95,
without community-correlated memberships; below 0.001 and NMI 1.00 with
them); between 40% and 60% it costs a few hundredths; below 40% the
level set is too sparse for the planted communities to be recovered
at any resolution, NMI falls to 0.4--0.7 and the loss to 0.2--0.6.
The membership spread has no detectable effect (correlation of loss
with σ: −0.03), which contradicts the earlier version's claim that
spread decides, and the Corollary 1 gap is a poor predictor of the loss
(correlation 0.43; 35 cells with a gap above 0.04 and a loss below
0.01, 12 cells with a gap below 0.01 and a loss above 0.1). When the
memberships carry community information the quartile cuts are nearly
free up to the upper quartile, because the surviving edges are
preferentially the internal ones, and binarising can then even help
recovery; when they carry none, the upper-quartile cut loses 0.17 and
drops NMI to 0.68. The absolute levels behave as the mean dictates: at
mean 0.9 every absolute cut is free, at mean 0.3 every absolute cut
above 0.4 destroys the partition. The scale-free statement a
practitioner can act on is therefore in terms of retained mass, not of
α, and not of spread. The constant-membership identity of Remark 3 was
checked on a planted graph with all memberships equal to c ∈ {0.25,
0.5, 0.75}: Q_F equals the crisp modularity of the same partition in
all three cases, binarising at α ≤ c is lossless, and the gap equals
γ(1/c − 1)Σ_κ a_κ b_κ to fifteen decimals.

Table 8 repeats the shortcut on the six real graphs. These rows were
computed in the earlier version at the single Algorithm-1 resolution
of Table 5 rather than with γ refitted per level set, and they are
retained with that caveat; the released script now refits γ per level
set and reports NMI between Π_α and Π_fuzzy on every graph, and the
rows are to be replaced by that output before submission. \\[CẦN CHẠY:
`python scripts/benchmark_binarisation.py --edges data/graphs/*_edges.csv`
rồi thay Bảng 8 bằng kết quả có γ refit; hiện giữ số của bản cũ.\\]""")
rep("""**Table 6.** The binarise-then-partition shortcut on the six real graphs
at the resolution γ₁ of Table 5.""", """**Table 8.** The binarise-then-partition shortcut on the six real graphs
at the single resolution γ₁ of Table 5 (earlier version; see text).""")
rep("""Three things are visible. While the cut retains most of the mass the
shortcut is nearly free: at α = 0.2 the level set is the whole support
graph, since the construction prunes at τ_E = 0.2, and the crisp
partition evaluated on the fuzzy graph is within 0.003 of the fuzzy
optimum on every graph; at α = 0.4 it is still within 0.004 on the
item-level graphs and within 0.004 on the KC-level ones. Once the cut
discards most of the mass the shortcut is destructive, and the two
evaluations of the same partition pull apart: at α = 0.6 the 2009
item-level level set keeps 52 of 4950 edges, crisp Leiden reports Q_α =
0.985 on it, and the same partition scores 0.035 on the fuzzy graph, a
loss of 0.93 against the fuzzy optimum; on the 2017 item-level graph the
loss is 0.21, and on the three KC-level graphs the level-set partitions
are worse than the already negative fuzzy optimum by 0.04 to 0.07. The
membership spread decides which regime applies: on MOOCCubeX-CS, where
the mean membership is 0.934, every cut up to α = 0.8 keeps at least 86
of 91 edges and the loss never exceeds 0.010, while on the ASSISTments
graphs, with mean memberships 0.42--0.47, no edge survives α = 0.8 at
all. Alongside, the identity of Corollary 1 was evaluated on every graph
at the same resolution: the gap Q_F − ∫Q̃ equals γ Σ_κ Cov(f_κ, g_κ) to
four decimals, ranges from 0.004 (MOOCCubeX-CS) through 0.040--0.053
(the ASSISTments KC-level and 2009 item-level graphs) to 0.339 (2017
item level), and the Cauchy--Schwarz bound of the corollary is within
0.001 of the gap on all six graphs, so the reportable bound is tight in
practice. The gap and the shortcut cost measure different things, which
Section 10.3 returns to: the 2009 item-level graph has a gap of 0.042
and a shortcut loss of 0.93 at α = 0.6, because the gap concerns one
partition evaluated two ways while the shortcut changes the partition.""",
"""The real graphs tell the same story in retained mass. At α = 0.2 the
level set is the whole support graph, since the construction prunes at
τ_E = 0.2, and the crisp partition is within 0.003 of the fuzzy optimum
on every graph; at α = 0.4 it is within 0.004 on all six. At α = 0.6
the 2009 item-level level set keeps 52 of 4950 edges, crisp Leiden
reports Q_α = 0.985 on it, and the same partition scores 0.035 on the
fuzzy graph; part of that 0.93 is a resolution artefact of reusing the
fuzzy-graph γ on a level set of very different density, which the
refitted rows will separate from the loss proper. On MOOCCubeX-CS,
where the mean membership is 0.934, every cut up to α = 0.8 keeps at
least 86 of 91 edges and the loss never exceeds 0.010, while on the
ASSISTments graphs, with mean memberships 0.42--0.47, no edge survives
α = 0.8 at all; these are statements about the mean, not about the
spread. The Corollary 1 identity was evaluated on every graph at the
same resolution, the gap equalling γ Σ_κ Cov(f_κ, g_κ) to four
decimals and ranging from 0.004 (MOOCCubeX-CS) to 0.339 (2017 item
level); the Cauchy--Schwarz bound lies within 0.001 of it on all six
graphs because the graphs are near-reciprocal (up to 0.998), so f_κ ≈
g_κ and the bound is an identity up to the asymmetry. The gap of 0.042
on the 2009 item-level graph next to a shortcut loss of 0.93 is the
real-data instance of Remark 3.""")

# ---------------------------------------------------------- Section 9.2
rep("""A log-log fit of runtime against edge count from 250 nodes upward gives
empirical exponents of 1.48 and 1.49 in the two regimes, above the
O(\\|E\\| log \\|C\\|) per-iteration bound because the number of
communities, and hence the number of aggregation levels traversed,
itself grows roughly linearly in \\|C\\| on these graphs; the
per-iteration cost is consistent with the bound.""",
"""A log-log fit of runtime against edge count from 250 nodes upward gives
empirical exponents of 1.48 and 1.49 in the two regimes, above the
O(\\|E\\| log \\|C\\|) per-iteration bound. The number of communities grows
roughly linearly in \\|C\\| on these graphs (Table 9), and we attribute
the excess to the number of aggregation levels traversed; the released
script records that number per run so the attribution can be checked.
The density of these graphs, about 1.5 edges per node, is that of the
curated graph and not of the KC-level graphs (about 50), so Table 9
says nothing about scaling in the dense regime.""")
rep("""**Table 7.** Fuzzy Leiden on synthetic fuzzy graphs, γ = 0.5, mean of 5
seeds.""", """**Table 9.** Fuzzy Leiden on synthetic fuzzy graphs, γ = 0.5, mean of 5
seeds.""")
rep("""What Table 7 supports is a claim about the optimiser, not about real
graphs of that size:""", """What Table 9 supports is a claim about the optimiser, not about real
graphs of that size:""")
rep("""Lemma 1(i) was checked on the seed-0 partition of every size and regime,
3523 communities in total across the twelve configurations, with zero
counterexamples; the released script records the check per
configuration.""", """Lemma 1(i) was checked on the seed-0 partition of every size and regime,
3523 communities in total across the twelve configurations, and was not
violated; the released script records the check per configuration.""")

# ---------------------------------------------------------- Section 9.3
rep("""**Table 8.** Partitioners on the real MOOCCubeX-CS graph, all evaluated
under Q_F (γ = 0.5), best of 10 seeds each.""", """**Table 10.** Partitioners on the real MOOCCubeX-CS graph, all evaluated
under Q_F (γ = 0.5), best of 10 seeds each.""")
rep("""**Partitioners.** Table 8 compares""", """**Partitioners.** Table 10 compares""")
rep("""All three methods land within 0.04 of each other on this small graph,
with Fuzzy Leiden ahead;""", """All three methods land within 0.04 of each other on this graph of 59
nodes and 91 edges, which Section 9.2 identifies as the unstable
regime, so the comparison is indicative only; Fuzzy Leiden is ahead;""")
rep("""**Composition operators.** Table 9 sweeps both parametric families on
the real ASSISTments 2009 item-level fuzzy graph of Table 3 (371 nodes,
4950 directed fuzzy edges, mean membership 0.421) over 1529 ordered node
pairs reachable within four hops from 60 sampled sources, taking for
each pair the best path under each operator.""",
"""**Composition operators.** Table 11 sweeps both parametric families on
the real ASSISTments 2009 item-level fuzzy graph of Table 3 (371 nodes,
4950 directed fuzzy edges, mean membership 0.421) over 1529 ordered node
pairs reachable within four hops from 60 sampled sources, taking for
each pair the exact best path under each operator. The best paths were
computed by bounded enumeration in the earlier version and are exact
within the four-hop limit; the released script now computes them by
the generator-based shortest path of Section 7.3, which the enumeration
agrees with to 10\\^{−15} on every family and every target checked, and
without a hop limit, so the four-hop truncation can be dropped.""")
rep("""**Table 9.** Parametric t-norm families and compensatory alternatives on
the real ASSISTments 2009 item-level fuzzy graph, 1529 reachable pairs.""",
"""**Table 11.** Parametric t-norm families and compensatory alternatives
on the real ASSISTments 2009 item-level fuzzy graph, 1529 reachable
pairs within four hops (earlier computation; the released script
recomputes the table without the hop limit and with per-source Kendall
τ-b and bootstrap intervals). \\[CẦN CHẠY: `python
scripts/exact_tnorm_paths.py data/graphs/assist2009_item_edges.csv` và
cập nhật cột ρ bằng τ-b theo nguồn kèm khoảng tin cậy bootstrap.\\]""")
rep("""First, the operator choice is consequential on this data, and the
minimum has to be argued for rather than defaulted to. Rank agreement
ranges from 0.529 (Łukasiewicz, equivalently Yager p = 1 and
Schweizer--Sklar p = 1) to 0.952 (Yager p = 10), so adopting Łukasiewicz
would reorder the induced ranking substantially. The mean edge
membership is what drives this: at a mean membership of 0.421 the strict
t-norms have a great deal of room to erode, and erosion reaches 0.574,
with Łukasiewicz annihilating almost every multi-hop path outright. The
same sweep on the MOOCCubeX-CS graph, whose mean membership is 0.934,
gives rank agreement never below 0.931 and maximum erosion 0.037 over
316 reachable pairs, which is worth recording as the complementary
caution: the operator choice is consequential exactly where the
membership distribution is wide, and nearly free where it is
concentrated.""",
"""First, the operator choice is consequential on this data. Pooled rank
agreement ranges from 0.529 (Łukasiewicz, equivalently Yager p = 1 and
Schweizer--Sklar p = 1) to 0.952 (Yager p = 10). Two cautions apply to
that statistic. It pools pairs across sources and hop lengths, so it is
partly a hop-length and tie effect: Łukasiewicz returns 0 for most 3-
and 4-hop pairs, and the ordering Equation (16) consumes is within a
source, so the per-source Kendall τ-b the released script now reports
is the relevant quantity. And the comparison with MOOCCubeX-CS, whose
mean membership is 0.934 and where agreement never falls below 0.931
over 316 reachable pairs, does not identify the mean as the cause,
since the two graphs differ in size, density, acyclicity and hop-length
distribution as well; a mean near 1 compresses every Archimedean
operator toward the minimum regardless of spread, which is an algebraic
fact rather than a measured one, and the earlier version's phrase
"depends almost entirely on the spread" conflated mean with spread. The
controlled statement is Proposition 3 and its verification below: the
reordering is a consequence of the operators' sensitivity to the
membership scale, and it is absent for max-min by construction.""")
rep("""Third, only the minimum has the optimal-substructure property that makes
best-path computation a Dijkstra variant with the O(\\|E\\| log \\|C\\|)
bound of Section 3.1. Under a strict t-norm the best path is not
composed of best sub-paths and exact computation requires enumeration,
which is what was done for Table 9 and why the sweep runs over four-hop
paths from a sample of sources. The gap is measured: on the 371-node
item-level graph, max-min best paths from the 60 sources to every
reachable node complete in 8 ms, while the four-hop enumeration behind
Table 9 expands 8.3 million partial paths in 2.7 s, a factor of roughly
350, and the enumeration grows as the mean out-degree to the path-length
power. Adopting a strict t-norm as the default would trade a measured
O(\\|E\\| log \\|C\\|) cost for an exponential one; the families remain a
documented configuration option in the released code. Table 10 records
the same comparison for the three standard operators on the sparse
MOOCCubeX-CS graph, where the mean membership is 0.934 and the picture
is correspondingly different; all 316 pairs reachable within five hops
are used.""",
"""Third, the cost of the alternatives is that of a shortest-path sweep,
not of an enumeration. On synthetic graphs of 371 nodes at 1.5, 4 and 8
edges per node, Dijkstra on the generator lengths from ten sources
takes 0.73×, 0.39× and 0.26× the time of the max-min widest-path sweep,
the four-hop Bellman--Ford 1.7×, 0.28× and 0.20×, while the enumeration
of the earlier version grows from 0.3× to 1.7× to 15× as the density
rises (29, 3207 and 44015 partial paths expanded). The "350×" of the
earlier version was the cost of our enumeration on the denser real
graph. The ordinal-invariance claim of Proposition 3 was verified on
the 371-node synthetic graph under ten random strictly increasing
piecewise-linear rescalings of μ_E from twenty sources: Kendall τ
between the orderings of reachable targets before and after rescaling
is 1.000 for max-min on every source and map, and falls to a mean of
0.97 (minimum 0.33) for the product, 0.90 (0.33) for Łukasiewicz, 0.88
(0.33) and 0.98 (0.89) for Yager p = 0.5 and 2, and 0.99 (0.94) for
Schweizer--Sklar p = −2. Table 12 records the comparison for the three
standard operators on the sparse MOOCCubeX-CS graph, where the mean
membership is 0.934 and the picture is correspondingly compressed; all
316 pairs reachable within five hops are used.""")
rep("""**Table 10.** Mean fuzzy path strength by hop count, real MOOCCubeX-CS
graph, 316 reachable pairs.""", """**Table 12.** Mean fuzzy path strength by hop count, real MOOCCubeX-CS
graph, 316 reachable pairs.""")
rep("""Neither behaviour is
unconditionally correct: the minimum answers how strong the weakest
prerequisite on a path is, which is the question Equation (16) is built
around, while the strict t-norms answer how much cumulative erosion a
long chain suffers.""", """Neither behaviour is
unconditionally correct: the minimum answers how strong the weakest
prerequisite on a path is, which is the question Equation (16) is built
around and which is well posed on an ordinal scale, while the
Archimedean t-norms answer how much cumulative erosion a long chain
suffers, which presupposes a cardinal one.""")

# ---------------------------------------------------------- Section 9.4
rep("""**Table 11.** Cut-point sensitivity of the fuzzification, 505-point grid
per setting.""", """**Table 14.** Cut-point sensitivity of the fuzzification, 505-point grid
per setting.""")
rep("""Table 11 reports the L¹ distance""", """Table 14 reports the L¹ distance""")
rep("""The decision is considerably more robust than the membership. The worst
single shift of ±0.10 changes the defuzzified band at 6.3% of grid
points and the worst joint shift at 13.9%.""", """The defuzzified band is more robust than the membership: the worst
single shift of ±0.10 changes it at 6.3% of grid points and the worst
joint shift at 13.9%. The gate of Section 8.3 tests μ_par against
τ_ZPD rather than taking the argmax, so band flips are not gate flips;
the released script reports both.""")
rep("""What Table 11 bounds is the input-side perturbation, not its downstream
propagation.""", """What Table 14 bounds is the input-side perturbation, not its downstream
propagation.""")

# ---------------------------------------------------------- Section 9.5
old95 = s[s.index("### 9.5 Interval type-2 state versus the type-1 collapse"):s.index("### 9.6 Component latency")]
new95 = """### 9.5 Gates on the uncertain state

The evaluation grid is 201 point estimates spanning \\[0,1\\] crossed
with seven half-widths ησ ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30},
giving 1407 states; the grid is the author's choice, and the
percentages below are properties of it, not of any population of
learners. The footprint of Equation (13) is well formed on every
state (no μ̲ \\> μ̄, no membership outside \\[0,1\\], a point at ησ =
0), and the cautious collapse lies inside its own footprint for μ_unk
and μ_mas everywhere and for μ_par on 92.7% of states.

**Table 13.** Pass rate of five gates at τ_ZPD = 0.4 by half-width,
201 point estimates per row. IT2 is the three-valued gate on the raw
bounds of Equation (13); the earlier version of this table normalised
the bound vectors separately before gating, which can make the pass
and fail tests fire on the same state, and is withdrawn.

  ---------------------------------------------------------------------------------------
  ησ      Collapse  Interval-image  Point       Probability  Expected    IT2 pass / indet. / fail
                    type-1 (ii)     estimate    of ZPD (iii) membership
                                    (i)                      (iv)
  ------- --------- --------------- ----------- ------------ ----------- --------------------------
  0.00    27.4%     27.4%           27.4%       27.4%        27.4%       27.4 / 0.0 / 72.6

  0.02    22.9%     22.9%           27.4%       27.4%        27.4%       22.9 / 8.5 / 68.7

  0.05    16.9%     16.9%           27.4%       26.4%        27.4%       16.9 / 20.4 / 62.7

  0.10    0.0%      7.5%            27.4%       26.4%        27.4%       7.5 / 39.3 / 53.2

  0.15    0.0%      0.0%            27.4%       23.4%        28.4%       0.0 / 56.7 / 43.3

  0.20    0.0%      0.0%            27.4%       1.5%         28.9%       0.0 / 67.2 / 32.8

  0.30    0.0%      0.0%            27.4%       0.0%         0.0%        0.0 / 84.6 / 15.4
  ---------------------------------------------------------------------------------------

The cautious collapse admits nothing from ησ = 0.10, as h\\^\\* = 0.075
predicts. The interval-image type-1 gate, which is the "pass" rule of
the IT2 gate, admits the same states as the collapse up to ησ = 0.05
and 7.5% at 0.10, then nothing; the IT2 gate therefore does not avoid
rejection at large widths, it relabels part of it as indeterminate,
and that is the whole of what the type-2 form adds. The
point-estimate gate admits 27.4% at every width because it ignores σ;
the probability gate and the expected-membership gate keep admitting
to ησ = 0.20 and degrade only at 0.30. No gate in the table is
validated against an outcome, so the table says which constructions
are cautious by design and which are not, and nothing about which is
right. Constructing the footprint costs 0.09 ms per 1407 states
against 0.03 ms for the collapse, which is negligible.

"""
s = s.replace(old95, new95)

# ---------------------------------------------------------- Section 9.6
rep("""the 8-pass Monte-Carlo-dropout encoder of Section 8.2 over a
20-step history 79.7 ms (online, mean of 20 runs); footprint
construction for 1407 states 0.09 ms (Section 9.5). The optimiser and
the fuzzification are therefore not where a deployed pipeline spends its
time; the encoder is. \\[CẦN XÁC MINH: mô hình mã hóa dùng để đo 79.7 ms
đã được huấn luyện trên tập nào, hay chỉ là mô hình khởi tạo để đo thời
gian; ghi rõ một trong hai.\\]""",
"""an 8-pass Monte-Carlo-dropout forward pass of the encoder architecture
of Section 8.2 over a 20-step history 79.7 ms (online, mean of 20 runs;
the timing was taken on an initialised network of the stated
architecture, since the forward cost does not depend on the weights,
and no trained encoder is evaluated in this paper); footprint
construction for 1407 states 0.09 ms (Section 9.5). The optimiser and
the fuzzification are therefore not where a deployed pipeline spends
its time; the encoder is.""")

# ---------------------------------------------------------- Discussion
old101 = s[s.index("### 10.1 What the three costs turn out to be"):s.index("### 10.2 What is out of scope")]
new101 = """### 10.1 What the three costs turn out to be

The answers to the three-part question of Section 1 differ in kind,
and two of them changed between the earlier version of this paper and
this one.

For binarisation the loss is bounded in the sense of Proposition 2,
but the bound is not the loss: Remark 3 shows that the Corollary 1 gap
is positive on graphs that are lossless to binarise, and the planted
benchmark of Table 6 shows that what governs the measured loss is the
fuzzy mass the cut discards, with no detectable effect of the
membership spread. The practitioner's statement is therefore not
"report the gap" but "report the retained mass": a cut that keeps
more than about 60% of the fuzzy mass is free, one that keeps less than
40% destroys the partition at any resolution, and the quantiles of
μ_E, not absolute levels, are the scale-free way to choose a cut if one
must be chosen. The resolution threshold of Proposition 1 is a known
identity stated for fuzzy mass; its balanced form explains the
degenerate rows of Table 4 in order of magnitude, and the null-model
step added to Algorithm 1 is what separates a partition from a
near-singleton artefact, returning "no structure detected" on planted
graphs without communities and on a dense co-practice-like graph.

For the state the loss is a property of one construction. The cautious
collapse rejects every candidate once the half-width passes h\\^\\* =
½(1 − τ_ZPD) max(a, b); a type-1 gate on the interval image, a
probability gate, or the point estimate does not, and the interval
type-2 form adds a three-valued split rather than a larger pass set.
Which gate is right is an outcome question (Section 10.2).

For path composition the computational objection is withdrawn: best
paths under every Archimedean t-norm are shortest paths through the
additive generator, at a cost comparable to the max-min sweep. What
remains is semantic. Max-min is the only operator among those
considered whose best-path ordering survives every strictly increasing
rescaling of μ_E (Proposition 3), and since the construction of
Section 8.1 fixes the order of μ_E but not its unit, that is the
argument for it; the pooled rank agreements of Table 11 (0.53 to 0.95
on one graph, above 0.93 on another) measure how much the alternatives
move under one particular scale, and the per-source statistics with
bootstrap intervals that the released script now produces are the
form in which that measurement should be read.

Two findings are negative and are reported as such. The resolution
defaults do not transfer between graphs of different density, which is
why Table 1 refers the resolution to Algorithm 1 rather than to a
number. And the reading of μ_E as prerequisite strength, which Equation
(15) presumes, is well founded at the item level and questionable at
the skill level of the ASSISTments family, because co-practice is
near-universal there; the null-model test on those graphs, once run,
is the modularity-side form of that statement.

"""
s = s.replace(old101, new101)
rep("""This paper prices three shortcuts; it does not evaluate a recommender.
No comparison of predictive accuracy against knowledge-tracing models,
no path-quality comparison against a recommendation baseline and no
study of human-perceived explanation quality is reported, and the
retrieval and generation layer that would sit on top of Equation (16) is
deliberately left out, because any claim about it would require an
outcome study that the present data do not support. The graphs,
partitions and states produced here are released so that such a study
can be run against them.""",
"""This paper prices three shortcuts; it does not evaluate a recommender,
and for two of the three shortcuts the price is measured on proxies.
The third part of the question of Section 1, whether the cost changes a
decision, is answered with ground truth only for binarisation, on
planted partitions. For the state no trained estimator is evaluated:
the distribution of ησ on held-out learners, the calibration of σ, and
a selective-prediction comparison of the gates of Table 13 against
held-out outcomes are the experiment that would settle Section 6, and
they are not reported here. For composition no held-out transition or
success rate is predicted. No comparison of predictive accuracy against
knowledge-tracing models, no path-quality comparison against a
recommendation baseline and no study of human-perceived explanation
quality is reported, and the retrieval and generation layer that would
sit on top of Equation (16) is left out. The graphs, partitions and
states produced here are released so that those studies can be run
against them, and the title, abstract and conclusion are worded as
measurements of what was measured.""")
rep("""*Proposition 2 is one-sided and partition-wise.* It bounds the α-average
for a fixed partition and says nothing by itself about the partition a
crisp detector would return on a level set. Table 6 is the empirical
substitute on six graphs, and it shows that the two quantities can be
far apart: a gap of 0.042 next to a shortcut loss of 0.93 on the same
graph. A result relating the two optima analytically is not available.""",
"""*Proposition 2 is a normalisation statement.* It bounds the α-average
of a mass-weighted surrogate for a fixed partition, and Remark 3 shows
that its gap is not a loss. A scale-free comparison, with per-cut
normalisation or quantile levels, that would relate Q_F analytically to
what a crisp detector returns on a level set is not available; Tables 6
and 8 are the empirical substitute.

*The real graphs have no reference structure.* The five ASSISTments
graphs come from the author's own pipeline with hand-set parameters,
and no signal or parameter is validated against a reference
prerequisite structure; the MOOCCubeX-CS graph is expert-curated but
has 59 nodes, 91 edges and zero reciprocity. The planted benchmark
supplies ground truth for the binarisation question only. Building the
fuzzy graph from the MOOCCubeX behaviour log and scoring μ_E against the
expert pairs, and using the skill-builder problem-set membership as a
reference partition for the item-level graphs, are the two checks the
released code is prepared for and this paper does not report.

*Algorithm 1 is not compared with other resolution-selection methods.*
Stability across γ, resolution profiles \\[46\\], consensus clustering
\\[48\\] and resolution-limit-free objectives \\[47\\] address the same
problem; the null-model step is the only external check reported, and
a comparison on planted benchmarks with varying density is left open.""")
rep("""*The interval type-2 state is verified but not exercised end to end.*
Section 9.5 establishes that the footprint is well formed, that the
collapse has two quantified defects and that the repair is cheap. What
the indeterminate region does across a full learning session requires
the outcome study of Section 10.2.

*Statistical reporting.* The rank agreements of Table 9 are point
estimates over 1529 pairs from 60 sampled sources, and the seed
dispersions of Table 7 are standard deviations over five seeds; no
confidence intervals are reported for the Spearman coefficients, and a
reader wanting them can obtain bootstrap intervals from the released
path enumeration.""",
"""*The gates on the uncertain state are compared on a grid, not on
learners.* Section 9.5 establishes which gates are cautious by design.
No gate is evaluated against an outcome, σ is never checked for
calibration, and the grid percentages are not population statistics.

*Statistical reporting.* The rank agreements of Table 11 are pooled
point estimates over 1529 pairs from 60 sampled sources; the released
script now reports per-source Kendall τ-b with bootstrap intervals over
sources, and the seed dispersions of Table 9 are standard deviations
over five seeds. The uncertainty of μ_E itself, estimated from a
minimum of 20 co-occurrences, is not propagated to the partitions or
the path rankings; a bootstrap over learners yielding interval-valued
memberships is the natural extension and is not reported.""")

# ---------------------------------------------------------- Conclusion
old11 = s[s.index("## 11. Conclusion"):s.index("## CRediT authorship contribution statement")]
new11 = """## 11. Conclusion

Three shortcuts that discard the grades on a fuzzy knowledge graph have
been priced, and the prices are reported as what was measured.

Binarising before partitioning: the fuzzy-graph modularity exceeds the
α-average of its level-set surrogates under global normalisation by γ
times the summed covariance of the community strength profiles, and
that gap is not the cost of binarising, since it is positive on graphs
that are crisp up to scale. The cost, measured on planted partitions
with the resolution refitted per level set, is governed by the fuzzy
mass the cut discards and not by the membership spread: free above
about 60% retained mass, destructive below 40%. The threshold at which
a bipartition beats the trivial partition is the split gain of
generalised modularity stated for fuzzy mass; Algorithm 1 seeds the
resolution from its balanced form and, with a null-model test, returns
"no structure detected" where every partition is degenerate.

Collapsing an interval-valued state to a type-1 triple by the cautious
endpoint rule costs every recommendation past a half-width of ½(1 −
τ_ZPD) max(a, b); the defect belongs to that rule, not to type-1
representations, and the comparison of gates on a trained estimator is
left to an outcome study.

Replacing max-min by an Archimedean t-norm cannot buy compensation,
since the minimum is the upper envelope of the class, and it costs no
more to compute, since best paths under every such t-norm are shortest
paths through the additive generator. What it gives up is ordinal
invariance: max-min best paths survive every strictly increasing
rescaling of the membership scale and no Archimedean t-norm's do, and
on real graphs the reordering runs from negligible to substantial.

All of this was checked on six real fuzzy graphs from four public
corpora, on planted partitions and on synthetic graphs to 20000 nodes,
by one script that rebuilds the graphs and re-derives the reported
values; three negative findings, on the transfer of resolution
defaults, on the prerequisite reading of skill-level graphs, and on the
cautious collapse, were kept. For practitioners: report the share of
fuzzy mass a cut retains rather than the modularity gap; seed the
resolution from the cut fraction and test the result against a null
model; use max-min unless the membership scale is cardinal; and treat
a cautious type-1 collapse as a choice with a measurable failure width.

Two directions follow. Interval-valued edge memberships obtained by a
bootstrap over learners, with max-min and modularity extended to
intervals, would propagate the uncertainty that the data actually
carry and link the three sections. And a gate comparison on a trained,
calibrated estimator against held-out outcomes would answer the one
question Section 6 cannot.

"""
s = s.replace(old11, new11)

# ---------------------------------------------------------- back matter
rep("""reproduces Tables 2 and 4 to 10
together with the numerical verification of Proposition 2 and Corollary
1, and compares 114 printed values against its own output, is provided
as supplementary material and will be maintained on a public repository.""",
"""reproduces every table, runs the planted benchmark of Table 6, the
null-model test of Table 7, the generator-based best paths and the
ordinal-invariance check of Section 9.3 and the gate comparison of Table
13, and compares the printed values listed in the supplement against
its own output, is maintained at
https://github.com/Bangth2026/fuzzy-kg-modularity.""")
rep("""\\[2\\] X. Cheng, Z. Zhang, J. Wang, L. Fang, C. He, Q. Guan, S. Pan, W.
Luo, Education-oriented graph retrieval-augmented generation for
learning path recommendation, arXiv:2506.22303 (2025). \\[CẦN XÁC MINH:
kiểm tra lại danh sách tác giả và phiên bản arXiv trước khi nộp\\]""",
"""\\[2\\] X. Cheng, Z. Zhang, J. Wang, L. Fang, C. He, Q. Guan, S. Pan, W.
Luo, GraphRAG-induced dual knowledge structure graphs for personalized
learning path recommendation, Proc. AAAI Conf. Artif. Intell. 40 (2026)
14610--14620. https://doi.org/10.1609/aaai.v40i17.38479""")
s = s.rstrip() + """

\\[44\\] M.E.J. Newman, Analysis of weighted networks, Phys. Rev. E 70
(2004) 056131.

\\[45\\] J. Reichardt, S. Bornholdt, Statistical mechanics of community
detection, Phys. Rev. E 74 (2006) 016110.

\\[46\\] A. Arenas, A. Fernández, S. Gómez, Analysis of the structure of
complex networks at different resolution levels, New J. Phys. 10 (2008)
053039.

\\[47\\] V.A. Traag, P. Van Dooren, Y. Nesterov, Narrow scope for
resolution-limit-free community detection, Phys. Rev. E 84 (2011)
016114.

\\[48\\] A. Lancichinetti, S. Fortunato, Consensus clustering in complex
networks, Sci. Rep. 2 (2012) 336.

\\[49\\] M. Gondran, M. Minoux, Graphs, Dioids and Semirings: New Models
and Algorithms, Springer, New York, 2008.

\\[50\\] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin,
J.V. Dillon, B. Lakshminarayanan, J. Snoek, Can you trust your model's
uncertainty? Evaluating predictive uncertainty under dataset shift, in:
Advances in Neural Information Processing Systems 32, 2019, pp.
13991--14002.
"""

# ---------------------------------------------------------- figure caption duplicate
dup = """Fig. 1. Footprint of uncertainty of the three linguistic labels induced
by the interval \\[p̄\\* − ησ, p̄\\* + ησ\\] at half-width ησ = 0.10,
cut-points (α_unk, α_par, α_mas) = (0.35, 0.60, 0.80). Shaded band:
lower and upper memberships of Equation (13). Dashed curve: the type-1
collapse of Equations (11)--(12). For the two shoulder labels the
collapse stays inside its footprint; for the middle label it leaves the
footprint on part of the axis, the phenomenon quantified over the full
grid in Section 9.5.

**Fig. 1.**"""
rep(dup, "**Fig. 1.**")

# ---------------------------------------------------------- misc cross-refs
rep("the phenomenon quantified over the full\ngrid in Section 9.5.](./media", "the phenomenon quantified over the full\ngrid in Section 9.5.](./media")
sub(r"which the measured grid brackets \(Table 12\)", "which the measured grid brackets (Table 13)")
sub(r"Section 9\.5 measures the\nindeterminate region, which grows with epistemic uncertainty to 37\.8% of\nthe state space\.",
    "Section 9.5 measures the\nindeterminate region on a grid.")
# Table 6 (planted) body
rows = R["rows"]
import collections, numpy as np
agg = collections.defaultdict(list)
for x in rows:
    agg[(x["mu_mean"], x["mu_sd"], x["shift"])].append(x)
cuts = ["alpha=0.4", "alpha=0.6", "alpha=0.8", "quantile=0.25", "quantile=0.5", "quantile=0.75"]
hdr = "  Mean  SD    Shift  Fuzzy (NMI; gap)  " + "  ".join(f"{c:<16s}" for c in ["α = 0.4", "α = 0.6", "α = 0.8", "q = 0.25", "q = 0.50", "q = 0.75"]) + "\n"
sep = "  ----- ----- ------ ----------------  " + "  ".join("-" * 16 for _ in cuts) + "\n"
body = ""
for key, xs in sorted(agg.items()):
    f = np.mean([x["fuzzy"]["NMI_truth"] for x in xs]); g = np.mean([x["fuzzy"]["gap_cor1"] for x in xs])
    line = f"  {key[0]:<5.1f} {key[1]:<5.2f} {key[2]:<6.1f} {f:.2f}; {g:.3f}       "
    cells = []
    for c in cuts:
        v = [x["cuts"][c] for x in xs if x["cuts"][c]]
        if len(v) < len(xs):
            cells.append(f"{'---':<16s}")
        else:
            cells.append(f"{np.mean([y['NMI_truth'] for y in v]):.2f}/{np.mean([y['loss'] for y in v]):.3f}/{np.mean([y['mass_kept'] for y in v]):.2f}".ljust(16))
    body += line + "  ".join(cells) + "\n\n"
table6 = "  " + "-" * (len(hdr) - 4) + "\n" + hdr + sep + body + "  " + "-" * (len(hdr) - 4) + "\n"
s = s.replace("@@TABLE6@@", table6)

assert "CẦN XÁC MINH" not in s, "draft notes remain"
open("ms_v2.md", "w", encoding="utf-8").write(s)
print("words:", len(s.split()), "remaining CẦN CHẠY markers:", s.count("CẦN CHẠY"))
