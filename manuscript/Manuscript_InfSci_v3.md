# What binarisation, type-1 collapse and non-idempotent t-norms cost on fuzzy knowledge graphs built from learner logs

**Author.** Hai Bang Truong

**Affiliation.** Saigon International University, Ho Chi Minh City,
Vietnam

**Corresponding author.** Hai Bang Truong, truonghaibang@siu.edu.vn

**ORCID.** 0000-0001-5094-3128

## Highlights

-   The Corollary 1 gap is a normalisation quantity, not the cost of
    binarising; the cost is measured instead
-   On planted partitions the cost of binarising tracks the fuzzy mass
    the cut discards, not the spread
-   A resolution threshold for general bipartitions, with a null-model
    test and a "no structure" outcome
-   Best paths under Archimedean t-norms are shortest paths; only
    max-min is ordinally invariant
-   Six graphs from four public corpora, planted benchmarks, one script
    that re-derives the reported values

## Abstract

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

**Keywords:** Fuzzy graphs; Community detection; Modularity; Resolution
limit; Triangular norms; Learner modelling

## 1. Introduction

A fuzzy knowledge graph built from behavioural logs is a directed fuzzy
graph in the sense of Rosenfeld \[1\]: its nodes are knowledge
components, its edges carry a membership μ_E ∈ \[0,1\] read as
prerequisite strength, and its nodes carry a graded learner state that a
stochastic estimator returns as a mean and a dispersion rather than a
number. Such graphs are now a standard substrate for learning-path
recommendation and for retrieval pipelines that summarise a curriculum
community by community \[2,3\]. Three operations consume them. The graph
is partitioned into communities so that each community can be summarised
and indexed; the learner state is passed through a gate that decides
whether a concept lies in the learner's zone of proximal development;
and candidate paths are scored by composing edge memberships along them.

In each of these three operations a cheap practice, which our own
earlier pipelines followed, discards the grades at the first
opportunity. The fuzzy graph is thresholded at some α and a crisp
community detector is run on the level set. The pair (p̄, σ) is collapsed
into one type-1 membership triple by evaluating each linguistic label at
the interval endpoint most cautious for it. Max-min composition, which
Rosenfeld's strength of connectedness makes canonical \[1,4\], is
replaced by a non-idempotent t-norm so that long paths are penalised.
Each shortcut is one line of code and produces output that looks
reasonable. None of them comes with a statement of what it costs.

That cost is the problem this paper studies. The question is the same
for all three shortcuts and has three parts. Is the information lost by
the shortcut bounded, in a form that can be proved rather than
illustrated? Is the bound computable from the graph and the partition
alone, so that a practitioner can report it next to the result? And is
it large enough to matter, measured against a known answer rather than
against another proxy? A shortcut whose cost is small can be kept; one
whose cost is large and measurable should be replaced, and the
replacement priced. We were able to answer the third part with ground
truth for the first shortcut, on planted partitions; for the other two
the evidence remains proxy-based, and Section 10.2 says so.

The first shortcut is the one that needs the most machinery. Modularity
\[5\] with its scalable optimisers Louvain \[6\] and Leiden \[7\] is
defined for non-negative edge weights, and a fuzzy graph is, in the
narrow sense, such a graph. The functional Q_F obtained by substituting
memberships for weights therefore inherits boundedness, exact
multi-level aggregation, monotone improvement and Leiden's connectivity
guarantee verbatim; Section 5.2 records these as a lemma with short
proofs, because a reader of a fuzzy-graph paper is entitled to see that
nothing breaks, not because anything new is being claimed. What is new
begins where the membership scale does mathematical work. Because μ_E is
bounded on \[0,1\] with fixed semantics, its level sets E_α form a
canonical family of crisp graphs, and Q_F can be compared with the
modularities of those level sets. Proposition 2 shows that Q_F dominates
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
non-trivial partition.

The second shortcut concerns the learner state. Converting an interval
\[p̄ − ησ, p̄ + ησ\] into memberships over *unknown*, *partial* and
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
output is left as the outcome study of Section 10.2.

The third shortcut has two answers, one of which corrects our earlier
version. Since T(a,b) ≤ min(a,b) for every t-norm T \[8\], no member of
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
alternatives reorder best paths on real graphs.

The graphs on which all of this is checked are real. MOOCCubeX \[3\]
ships an expert-curated prerequisite graph with continuous confidences.
Three public ASSISTments releases supply interaction logs from which
Section 8.1 constructs fuzzy graphs end to end at two granularities. A
single released script downloads nothing but these public files,
rebuilds every graph, reproduces every table and compares the printed
values listed in the supplement against its own output. The application supplies the graphs and
the semantics of the operations; this paper reports no comparison of
predictive accuracy against knowledge-tracing models and no study of
recommendation quality, and Section 10.2 says why.

**Contributions.**

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

Section 2 positions the work. Section 3 collects preliminaries and
Section 4 defines the fuzzy knowledge graph and the graded state.
Sections 5, 6 and 7 treat the three shortcuts in turn, Section 6 in the
form of a methodological remark. Section 8 states the construction
pipeline, Section 9 the experiments, and Sections 10 and 11 discuss and
conclude.

## 2. Related work and positioning

### 2.1 Community detection with graded information

Two distinct objects are called fuzzy in the community-detection
literature, and the distinction decides what this paper is and is not
about. The larger line fuzzifies the *partition*: Nepusz et al. \[9\]
and Havens et al. \[10\] let a vertex belong to several communities with
graded membership while the graph stays crisp, the soft-modularity and
fuzzy-modularity maximisers that followed optimise such membership
matrices by local iterative updates \[11,12,13,14\], the evidential
variant works over the power set of communities \[15\], and a recent
perspective article surveys the line as a whole \[16\]. The present
paper fuzzifies the *edges* and keeps the partition crisp, for a reason
stated in Section 5.1; the two generalisations act on different
arguments of the same functional and neither contains the other, and
Appendix A writes down the functional that fuzzifies both and locates
the algebraic price of doing so.

A second, smaller line keeps the graph crisp but enriches the objective
with a fuzzy measure. Barroso et al. \[17\] and Gutiérrez García-Pardo
et al. \[18\] add a flow-based capacity to Louvain-type optimisation on
directed graphs and report improved partitions on mobility networks.
That construction is the one Lemma 1(iii) shows to be incompatible with
exact multi-level aggregation: a non-additive measure over edge sets
cannot be contracted into self-loops, so the multi-level scheme becomes
a heuristic. The present paper stays with additive fuzzy mass precisely
to keep the optimiser exact, and treats the relation between Q_F and the
α-cuts, which has no analogue for a capacity-based objective, as its
main object.

Modularity with non-negative weights is Newman's weighted modularity
\[44\], and Q_F is that functional with memberships as weights.
Modularity at a fixed resolution has a known resolution limit \[19\];
the resolution parameter of Reichardt and Bornholdt \[45\] and the
resolution profiles of Arenas et al. \[46\] address it by scanning,
Traag et al. \[47\] by a resolution-limit-free objective, and
consensus clustering \[48\] by aggregating many runs. The split gain
behind Proposition 1 is the local-moving gain of generalised modularity
that Louvain and Leiden evaluate; what Section 5.3 does is state it for
a bipartition of a fuzzy graph, read off the threshold, and use it to
seed a resolution. We make no claim of novelty for the identity. The
interaction of resolution with density is documented in \[20\].

### 2.2 Interval-valued states and type-2 fuzzy sets

Interval type-2 fuzzy sets \[21\] and their type-reduction methods
\[22,23,24\] are mature, and type-2 fuzzy logic has been used in
adaptive teaching \[25\]. The question treated in Section 6 is narrower
and, as far as we know, has not been asked: when the interval comes from
a Bayesian dispersion, Monte-Carlo dropout in the present case \[26\],
is the usual cautious type-1 collapse an embedded set of the interval
type-2 membership the interval induces? Lemma 2 connects the interval to
the probability--possibility transformations of Dubois et al. \[27\] so
that the widening factor acquires a semantics, and the two defects of
the collapse are then exhibited constructively.

### 2.3 Path composition on fuzzy graphs

Max-min composition and Rosenfeld's strength of connectedness are the
classical answers to path strength on a fuzzy graph \[1,4,28\]. The
t-norm bound T ≤ min is standard \[8\], and the Yager \[29\] and
Schweizer--Sklar \[30\] families are textbook objects. Best paths under max-min and under
generator-based compositions are instances of the algebraic path
problem over a dioid \[49\], which is why both reduce to
Dijkstra-type sweeps (Section 7.3). The contribution of Section 7 and
Section 9.3 is the ordinal-invariance argument for max-min
(Proposition 3) and the measured reordering the alternatives cause on
real fuzzy graphs.

### 2.4 Fuzzy knowledge graphs in learner modelling

Knowledge graphs with prerequisite edges underlie most learning-path
recommenders, and the recent GraphRAG-style pipelines build or refine
such graphs with language models and consume them community by community
\[2\]. Graded prerequisite strengths and three-band mastery semantics
have long been used in fuzzy student modelling \[31,25\]. The present
paper does not propose a recommender; it supplies, for the three places
where such a pipeline discards grades, a bound, a threshold and a
measurement, and it releases the graphs on which they were obtained.

## 3. Preliminaries

Throughout, \[0,1\] carries its usual order, and for a fuzzy set A
\[32\] on a universe X with membership μ_A we write A_α = {x ∈ X :
μ_A(x) ≥ α} for its α-cut, α ∈ (0,1\].

### 3.1 Fuzzy graphs and strength of connectedness

A *fuzzy graph* \[1\] on a vertex set C is a pair (C, μ_E) with μ_E : C
× C → \[0,1\]; the *support graph* supp(G_F) is the crisp graph on C
whose edges are the pairs with μ_E \> 0 \[33\]. We use directed fuzzy
graphs, so μ_E is not assumed symmetric. For a directed path π = (u_0,
..., u_L) the *strength* of π is min\_{j} μ_E(u\_{j−1}, u_j), and the
*strength of connectedness* from a to b is

$$\mu_{E}^{\infty}(a,b)\mspace{6mu} = \mspace{6mu}\max_{\pi \in \Pi_{ab}}\mspace{6mu}\min_{(u,v) \in \pi}\mu_{E}(u,v),\quad\quad\text{(1)}$$

the maximum being over the set Π_ab of directed paths from a to b
\[1,4\]. Equation (1) is the max-min composition of μ_E with itself
iterated to closure, and coincides with the widest-path (bottleneck)
problem on supp(G_F), solvable by a Dijkstra-type sweep in O(\|E\| log
\|C\|).

### 3.2 Triangular norms

A *triangular norm* (t-norm) is a binary operation T on \[0,1\] that is
commutative, associative, non-decreasing in each argument and satisfies
T(a,1) = a \[8\]. From the boundary condition and monotonicity, T(a,b) ≤
T(a,1) = a and likewise T(a,b) ≤ b, so

$$T(a,b)\mspace{6mu} \leq \mspace{6mu}\min(a,b)\quad\text{for every t-norm }T\text{ and all }a,b \in \lbrack 0,1\rbrack.\quad\quad\text{(2)}$$

The minimum is therefore the largest t-norm. Two parametric families are
used below. The Yager family \[29\] is

$$T_{p}^{Y}(a,b)\mspace{6mu} = \mspace{6mu}\max\{ 0,\mspace{6mu} 1 - \left\lbrack (1 - a)^{p} + (1 - b)^{p}\rbrack^{1/p} \right\},\quad p > 0,\quad\quad\text{(3)}$$

with T_1\^Y the Łukasiewicz t-norm T_L(a,b) = max(0, a+b−1) and T_p\^Y →
min as p → ∞. The Schweizer--Sklar family \[30\] is

$$T_{p}^{SS}(a,b)\mspace{6mu} = \mspace{6mu}\lbrack\max\{ 0,\mspace{6mu} a^{p} + b^{p} - 1\}\rbrack^{1/p},\quad p \neq 0,\quad\quad\text{(4)}$$

with T_p\^{SS} → min as p → −∞, T_p\^{SS} → ab as p → 0 and T_1\^{SS} =
T_L.

### 3.3 Ordered weighted averaging

An OWA operator \[34\] of dimension n with weight vector w =
(w_1,...,w_n), w_j ≥ 0, Σ_j w_j = 1, maps x ∈ \[0,1\]\^n to Σ_j w_j b_j
where (b_1,...,b_n) is x sorted in decreasing order. Its *orness* is
orness(w) = (n−1)\^{−1} Σ_j (n−j) w_j, equal to 1 for the maximum and 0
for the minimum. OWA operators are averaging, hence idempotent and
compensatory, and are not t-norms except in the degenerate case w =
(0,...,0,1).

### 3.4 Interval type-2 fuzzy sets

An interval type-2 (IT2) fuzzy set à on X is characterised by a lower
membership μ̲\_Ã and an upper membership μ̄\_Ã with μ̲\_Ã(x) ≤ μ̄\_Ã(x) for
all x; the region between them is the *footprint of uncertainty* (FOU)
\[21\]. A type-1 set A is an *embedded* type-1 set of à if μ̲\_Ã(x) ≤
μ_A(x) ≤ μ̄\_Ã(x) for all x. Type reduction maps an IT2 set to a type-1
set or to a crisp value; the Karnik--Mendel procedure \[22\] does so
iteratively, while the Nie--Tan reduction \[23\] takes the pointwise
average of the two bounds and costs one pass.

### 3.5 Modularity and Leiden

For a crisp undirected graph with non-negative edge weights A\_{uv},
total weight 2W = Σ\_{uv} A\_{uv}, node strengths s(u) = Σ_v A\_{uv} and
resolution γ \> 0, the (generalised) modularity of a partition Π is
Σ\_{uv} \[A\_{uv} − γ s(u)s(v)/2W\] δ(κ(u),κ(v)) / 2W \[5\]. For a
directed graph, Leicht and Newman \[35\] replace the null term by the
product of out- and in-strengths, s\^{out}(u) s\^{in}(v)/W, with W the
total directed weight. Modularity at a fixed resolution has a known
resolution limit \[19\]: the scale at which it detects structure
interacts with the size and density of the graph; for a current overview
of the community-detection field see \[20\]. Leiden \[7\] optimises
these objectives by alternating a local-moving phase, a refinement phase
and an aggregation phase, and guarantees that every returned community
induces a connected subgraph --- a property that Louvain \[6\] does not
have.

## 4. Fuzzy knowledge graphs and graded node states

### 4.1 The fuzzy knowledge graph

Let C = {c_1, ..., c_K} be a finite set of *knowledge components* (KCs).
A *fuzzy knowledge graph* is a directed fuzzy graph

$$G_{F} = \left( C,E,\mu_{E} \right),\quad E \subseteq C \times C,\quad\mu_{E}:E \rightarrow \lbrack 0,1\rbrack,\quad\quad\text{(5)}$$

in which μ_E(c_a, c_b) is the strength with which c_a is a prerequisite
of c_b. A crisp knowledge graph is recovered when μ_E ∈ {0,1}. Path
strength between KCs is the strength of connectedness μ_E\^∞ of Equation
(1), and the same symbol is used throughout. The reading is
weakest-link: a strong path exists exactly when some route has every hop
strong.

### 4.2 The graded node state and its interval form

For a learner s_i and a KC c the state is a triple on the 2-simplex Δ_3,

$$\varphi_{i}(c) = \left( \mu_{i}^{unk}(c),\,\mu_{i}^{par}(c),\,\mu_{i}^{mas}(c) \right),\quad\mu_{i}^{unk}(c) + \mu_{i}^{par}(c) + \mu_{i}^{mas}(c) = 1,\quad\quad\text{(6)}$$

giving the memberships of c in the linguistic labels *unknown*,
*partial* and *mastered* for that learner. The full state is Φ_i ∈
Δ_3\^K.

What an estimator supplies is not a point but a pair (p̄, σ): a point
estimate of mastery and a dispersion. Section 6 treats the conversion of
that pair into (6), and shows that the interval \[p̄ − ησ, p̄ + ησ\] it
induces is properly an IT2 membership over the three labels rather than
an input to a type-1 one.

### 4.3 The two fuzzy-set constraints

Given a learner state Φ_i, an anchor KC c\^anchor and a target KC c\^\*,
a path recommender returns a next KC c\^rec and a path π = (c\^anchor,
c_1, ..., c_L, c\^\*) with c_1 = c\^rec. Two constraints on that output
are fuzzy-set-theoretic and are the ones this paper analyses:

**(P1) Zone of proximal development.** μ\^{par}\_i(c\^rec) ≥ τ_ZPD and
μ\^{mas}\_i(c\^rec) \< α_mas.

**(P2) Prerequisite consistency.** μ_E(c\_{j−1}, c_j) ≥ τ_path for every
consecutive hop of π.

Everything else a deployed recommender adds on top of these two
constraints, retrieval radius, rationale generation and latency budget,
is engineering and is outside the scope of this paper; Section 8.3
states the single scoring rule through which (P1) and (P2) are consumed,
and nothing in Sections 5 to 7 depends on more than that.

## 5. Modularity for fuzzy graphs

### 5.1 The functional

A fuzzy knowledge graph is directed, so the functional is stated in
directed form from the outset, following the treatment of directedness
in \[35\]. For G_F and a crisp partition Π = {κ_1, ..., κ_M} of C define
the *fuzzy-graph modularity*

$$Q_{F}(\Pi)\mspace{6mu} = \mspace{6mu}\frac{1}{W}\sum_{u,\, v\, \in \, C}^{}\left\lbrack \mu_{E}(u,v)\mspace{6mu} - \mspace{6mu}\gamma\,\frac{s^{out}(u)\, s^{in}(v)}{W} \right\rbrack\delta\left( \kappa(u),\kappa(v) \right),\quad\quad\text{(7)}$$

with the convention μ_E(u,v) = 0 for (u,v) ∉ E, where W = Σ\_{(u,v) ∈ E}
μ_E(u,v) is the total fuzzy mass, s\^{out}(u) = Σ_v μ_E(u,v) and
s\^{in}(u) = Σ_v μ_E(v,u) are the fuzzy out- and in-strengths of u, γ \>
0 a resolution parameter and δ the Kronecker delta. Writing W_κ for the
fuzzy mass internal to κ, S_κ\^{out} = Σ\_{u ∈ κ} s\^{out}(u) and
S_κ\^{in} = Σ\_{u ∈ κ} s\^{in}(u), so that Σ_κ S_κ\^{out} = Σ_κ
S_κ\^{in} = W, Equation (7) is equivalently

$$Q_{F}(\Pi)\mspace{6mu} = \mspace{6mu}\sum_{\kappa \in \Pi}^{}\left\lbrack \frac{W_{\kappa}}{W}\mspace{6mu} - \mspace{6mu}\gamma\,\frac{S_{\kappa}^{out}\, S_{\kappa}^{in}}{W^{2}} \right\rbrack.\quad\quad\text{(8)}$$

When μ_E is symmetric and each unordered pair is carried as two opposite
directed edges of equal membership, s\^{out} = s\^{in} = s, W is twice
the undirected mass W_u, and (8) reduces to the familiar undirected form
Σ_κ \[W_κ,u/W_u − γ (S_κ/2W_u)²\]; the undirected functional is
therefore the symmetric special case of (7), and no property below
depends on symmetry.

*Fuzzy Leiden* is the Leiden procedure of \[7\] with Q_F as the
objective optimised in the local-moving phase.

**Remark 1 (what is inherited and what is not).** Since μ_E is a
non-negative weight function, Equation (7) is algebraically an instance
of weighted directed modularity, and every property of that functional
that depends only on the weights being non-negative is inherited
verbatim. Lemma 1 lists those properties and proves them in a few lines
each, so that the inheritance is checked rather than assumed. What is
not inherited is anything in which the \[0,1\] scale of μ_E does work.
Three features separate a membership from a physical weight. The scale
is bounded with fixed semantics, μ_E = 1 asserting a full prerequisite,
so the level sets E_α = {e : μ_E(e) ≥ α} form a canonical family of
crisp graphs; an unbounded weight has no such family without an
arbitrary normalisation. The same μ_E enters two calculi, additively in
(7) and ordinally through the t-norm composition of Section 7, where
only its order on \[0,1\] matters. And μ_E is produced by the
orness-calibrated OWA aggregation of Section 8.1, so its value is an
evidence-pooling degree rather than a measurement. Section 5.4 is where
the bounded scale does mathematical work: it relates Q_F to the
modularities of the level sets, which is the comparison a practitioner
who binarises the graph is implicitly making.

**Remark 2 (relation to fuzzy modularity for fuzzy partitions).** Nepusz
et al. \[9\] and Havens et al. \[10\] fuzzify the *partition*: the graph
stays crisp and each vertex carries a membership vector over
communities, so that Newman modularity is generalised by replacing
δ(κ(u),κ(v)) with Σ_m u\_{um} u\_{vm}. Equation (7) fuzzifies the
*edges* and keeps δ. The two generalisations act on different arguments
of the same functional, both reduce to Newman modularity in their
respective crisp limits, and neither contains the other; the same
structural relation holds with the evidential line, in which the
partition is a credal one over the power set of communities while the
graph again stays crisp \[15\], and with the more recent algorithmic
work on fuzzy-partition community detection \[13,14,16\]. The partition
is kept crisp here for a stated reason rather than by default: the
downstream use of Section 8.3 attaches one summary and one index entry
to each community, so it consumes disjoint units, and the
crisp-partition local moving is what the scalability measurements of
Table 7 are about. A functional fuzzifying both arguments simultaneously
--- graded edges and graded community memberships --- is written down
and analysed in Appendix A: its aggregation step is exact only along
constant-membership blocks (Proposition A.1), which is the property
Section 10.3 identifies as decisive.

### 5.2 Inherited properties of the functional

**Lemma 1 (inherited properties).** *Let G_F = (C, E, μ_E) be a directed
fuzzy graph with μ_E : E → (0,1\], W \> 0 and γ \> 0, and let Fuzzy
Leiden be the procedure of \[7\] with Q_F as the local-moving objective.
Then:*

*(i) Well-connectedness. Every community returned at convergence induces
a weakly connected subgraph of the support graph supp(G_F).*

*(ii) Boundedness. Q_F(Π) ∈ \[−γ, 1\] for every partition Π of C.*

*(iii) Aggregation-invariance. Let G_F/Π be the contracted fuzzy graph
in which each κ ∈ Π becomes a node carrying a self-loop of mass W_κ and
inheriting out-strength S_κ\^{out} and in-strength S_κ\^{in}. For every
partition Π′ of the contracted node set, Q_F\^{G_F/Π}(Π′) =
Q_F\^{G_F}(Π′ ∘ Π), where Π′ ∘ Π is the induced partition of C.*

*(iv) Monotone improvement. Along the local-moving, refinement and
aggregation phases the sequence of Q_F values at the corresponding
partitions of C is non-decreasing, and increases strictly at every
accepted single-node move.*

*(v) Finite convergence. The procedure halts after finitely many
accepted moves at a partition that is a local optimum of Q_F with
respect to single-node moves, at which the γ-separation and
γ-connectivity guarantees of \[7\] hold.*

*(vi) Reduction. If μ_E ∈ {0,1} on E and γ = 1, Q_F is the directed
modularity of Leicht and Newman \[35\]; if additionally μ_E is
symmetric, it is Newman modularity; for general μ_E and γ = 1 it is the
weighted directed objective computed with weights μ_E.*

*Proof.* (i) The connectivity guarantee of \[7\] (guarantee 2,
γ-connectivity, with the proof in the supplementary information of that
paper) is a property of the refinement phase, which grows
sub-communities by a randomised locally greedy process over the edge
structure of the input graph and requires of the input only non-negative
edge weights. Substituting Q_F in the local-moving phase changes which
merges are proposed, not the growth process that acts on supp(G_F) once
they are proposed, and the growth process ignores edge orientation,
which is why the conclusion is weak connectivity. (ii) The first sum in
(8) lies in \[0,1\] because 0 ≤ Σ_κ W_κ ≤ W. With a_κ = S_κ\^{out}/W and
b_κ = S_κ\^{in}/W, both vectors lie on the probability simplex, so 0 ≤
Σ_κ a_κ b_κ ≤ 1, whence −γ ≤ Q_F ≤ 1. (iii) Contraction preserves the
total mass, assigns every internal edge to some W_κ or retains it
between contracted nodes, and carries S_κ\^{out} and S_κ\^{in} to the
contracted node by construction; Q_F depends on these quantities alone,
so the two expressions agree term by term. (iv) Local moving accepts a
move only on a strict increase, refinement merges only on a
non-decrease, and aggregation leaves the value unchanged by (iii). (v)
Partitions of a finite set are finitely many and by (iv) none is
revisited; at halting no single-node move improves Q_F. The guarantees
of \[7\] are stated for either modularity or CPM with non-negative
weights and apply. (vi) Substitute μ_E(u,v) = A\_{uv} in (7). ∎

Part (i) is a transfer statement: the guarantee of \[7\] applies once
the input is recognised as an instance of the weighted setting, and
Section 9.1 confirms it computationally with zero counterexamples; no
new theorem about fuzzy graphs is claimed. Weak connectivity is also the
right notion for this graph class, since a prerequisite graph is close
to acyclic and strong connectivity of a community would assert mutual
prerequisite reachability among its concepts. Part (iii) is the
load-bearing one for the optimiser and rests on the additivity of the
mass assignment A ↦ Σ\_{e ∈ A} μ_E(e): were internal mass a non-additive
fuzzy measure of the edge set, a Choquet or Sugeno integral with respect
to a capacity ν \[36\] as in the flow-capacity line of \[18\],
contraction would fail because ν(A ∪ B) need not equal ν(A) + ν(B) for
disjoint edge sets, and the multi-level scheme would degrade from an
exact reformulation to a heuristic. Additivity of the fuzzy mass is the
exact price of the exact optimiser.

### 5.3 The resolution threshold of a bipartition

**Proposition 1 (resolution threshold of a bipartition).** *Let B = {κ,
κ̄} be a bipartition of C with inter-community fuzzy mass c(B) (both
directions counted), and write a_1 = S_κ\^{out}/W, b_1 = S_κ\^{in}/W
and a_2, b_2 likewise for κ̄. Then Q_F(B) exceeds the value 1 − γ of the
trivial partition if and only if*

$$\gamma\mspace{6mu} > \mspace{6mu}\theta(B)\mspace{6mu} = \mspace{6mu}\frac{c(B)/W}{a_{1}b_{2} + a_{2}b_{1}}.$$

*For a strength-balanced split (a_i = b_i = 1/2) the threshold is θ(B)
= 2c(B)/W. Consequently no bipartition improves on the trivial partition
below γ = min_B θ(B), and 2c_bal/W, with c_bal the minimum balanced
cut, is an upper bound on that onset, not a lower bound: an unbalanced
block that is weakly attached to the rest can have θ(B) far below the
balanced value.*

*Proof.* For the trivial partition, (8) gives Q_F = 1 − γ. For B, Q_F =
(W − c)/W − γ (a_1 b_1 + a_2 b_2), and since (a_1 + a_2)(b_1 + b_2) = 1
the improvement over the trivial partition is

$$\Delta Q_{F}\mspace{6mu} = \mspace{6mu} - \,\frac{c}{W}\mspace{6mu} + \mspace{6mu}\gamma\,\left( a_{1}b_{2} + a_{2}b_{1} \right),$$

which is positive exactly under the stated condition. The denominator is
maximised, at 1/2, by a_1 = b_1 = 1/2, so among splits of equal cut
mass the balanced one has the smallest threshold; a split with smaller
cut mass can nevertheless have a smaller threshold. ∎

ΔQ_F is the split gain that Louvain and Leiden evaluate in their
local-moving step \[6,7\], written for a single bipartition of a fuzzy
graph; the resolution limit \[19,45\] is its consequence for
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
            null draws N (default 20); separation z_min (default 3);
            effect floor ΔQ_min (default 0.10); reference resolution γ_ref (default 1)
    1:  B ← strength-balanced spectral bisection of the symmetrised adjacency of G_F
    2:  c/W ← inter-block fuzzy mass of B divided by the total fuzzy mass W
    3:  γ1 ← ρ1 · (2c/W);   γ2 ← ρ2 · γ1
    4:  repeat
    5:      run Fuzzy Leiden at γ1 once per seed in S
    6:      if every run returns a non-trivial partition and the pairwise
               adjusted Rand index across seeds is at least θ_ARI then break
    7:      γ1 ← g · γ1;   γ2 ← g · γ2
    8:  Q_obs ← best Q_F over S at the reference resolution γ_ref (default 1)
    9:  for i = 1..N: rewire G_F preserving in- and out-degrees, permute the
            memberships over the edges, run Fuzzy Leiden at γ_ref over S,
            record the best Q_F
    10: ΔQ ← Q_obs − mean of the N null values;  z ← ΔQ / their standard deviation
    11: if z < z_min or ΔQ < ΔQ_min: return "no structure detected"
    Output: resolutions (γ1, γ2) and the partition at γ1, or "no structure detected"

The null test is run at a reference resolution rather than at the
accepted γ₁, because on a graph without structure γ₁ ends far above 1,
where both the observed and the null partitions are near-singleton and
a z-score compares two degenerate numbers; and it carries an
effect-size floor, because on dense graphs the null standard deviation
is of order 10\^{−3}, so a z-score alone certifies differences of a few
thousandths of Q_F (Table 7). The loop of steps 4 to 7 is capped at
twelve doublings in the released implementation; no real graph in
Section 9 reached the cap. Symmetrising
the adjacency in step 1 does not bias the estimate on directed graphs,
since c/W is a mass quantity and orientation-free. Section 9.1 reports
the revised algorithm on planted graphs with and without structure, on
a dense co-practice-like graph, and on the six real graphs.

### 5.4 Relation to the α-cuts, and the cost of binarising

Equation (7) is defined on the fuzzy graph directly. The
representation-theoretic question is how its value relates to the
modularities of the crisp directed graphs G_α = (C, E_α) obtained by
thresholding, E_α = {(u,v) ∈ E : μ_E(u,v) ≥ α}. Write \|E_α ∩ κ²\| for
the number of α-cut edges internal to κ, deg_α\^{out}(u) and
deg_α\^{in}(u) for the α-cut out- and in-degrees, and S_κ\^{out}(α) =
Σ\_{u ∈ κ} deg_α\^{out}(u), S_κ\^{in}(α) = Σ\_{u ∈ κ} deg_α\^{in}(u).
Both profiles α ↦ S_κ\^{out}(α) and α ↦ S_κ\^{in}(α) are non-increasing,
since E_α shrinks as α grows. Define the *α-cut modularity under global
normalisation*

$$\widetilde{Q}(\alpha;\Pi)\mspace{6mu} = \mspace{6mu}\sum_{\kappa \in \Pi}^{}\left\lbrack \frac{\left| E_{\alpha} \cap \kappa^{2} \right|}{W}\mspace{6mu} - \mspace{6mu}\gamma\,\frac{S_{\kappa}^{out}(\alpha)\, S_{\kappa}^{in}(\alpha)}{W^{2}} \right\rbrack.\quad\quad\text{(9)}$$

**Proposition 2.** *For every fuzzy knowledge graph G_F with W \> 0,
every γ \> 0 and every partition Π,*

$$Q_{F}(\Pi)\mspace{6mu} \geq \mspace{6mu}\int_{0}^{1}\widetilde{Q}(\alpha;\Pi)\, d\alpha,$$

*with equality if and only if, for every κ ∈ Π, (S_κ\^{out}(x) −
S_κ\^{out}(y)) (S_κ\^{in}(x) − S_κ\^{in}(y)) = 0 for almost every pair
(x, y) ∈ \[0,1\]². In particular, equality holds whenever for each κ at
least one of the two profiles is almost everywhere constant, and hence
whenever μ_E is crisp.*

*Proof.* For t ∈ \[0,1\] the layer-cake identity t = ∫\_0\^1 1\[t ≥ α\]
dα gives, summing over edges,

$$W_{\kappa} = \sum_{(u,v) \in E \cap \kappa^{2}}^{}\mu_{E}(u,v) = \int_{0}^{1}\left| E_{\alpha} \cap \kappa^{2} \right|\, d\alpha,\quad s^{out}(u) = \int_{0}^{1}\deg_{\alpha}^{out}(u)\, d\alpha,\quad s^{in}(u) = \int_{0}^{1}\deg_{\alpha}^{in}(u)\, d\alpha,$$

and hence S_κ\^{out} = ∫\_0\^1 S_κ\^{out}(α) dα and S_κ\^{in} = ∫\_0\^1
S_κ\^{in}(α) dα. The first term of (8) is therefore exactly the
α-average of the first term of (9). For the second term put f_κ(α) =
S_κ\^{out}(α)/W and g_κ(α) = S_κ\^{in}(α)/W, so that S_κ\^{out}/W =
∫\_0\^1 f_κ and S_κ\^{in}/W = ∫\_0\^1 g_κ. The correlation identity

$$\int_{0}^{1}f_{\kappa}g_{\kappa}\, d\alpha\mspace{6mu} - \mspace{6mu}\left( \int_{0}^{1}f_{\kappa}\, d\alpha \right)\left( \int_{0}^{1}g_{\kappa}\, d\alpha \right)\mspace{6mu} = \mspace{6mu}\frac{1}{2}\int_{0}^{1}{\int_{0}^{1}\left( f_{\kappa}(x) - f_{\kappa}(y) \right)}\left( g_{\kappa}(x) - g_{\kappa}(y) \right)\, dx\, dy$$

holds for any pair of integrable functions on \[0,1\]. Since f_κ and g_κ
are both non-increasing, the integrand on the right is non-negative
pointwise, which is Chebyshev's integral inequality: (∫ f_κ)(∫ g_κ) ≤ ∫
f_κ g_κ, with equality if and only if the integrand vanishes almost
everywhere. Summing over κ and multiplying by −γ \< 0 reverses the
inequality, and adding the two terms gives the claim. If for each κ one
of the two profiles is a.e. constant, the integrand vanishes and
equality holds; if μ_E is crisp then E_α = E for all α ∈ (0,1\], both
profiles are constant, and equality holds. In the symmetric case f_κ =
g_κ, Chebyshev's inequality reduces to Jensen's inequality for the
square, (∫ f_κ)² ≤ ∫ f_κ². ∎

**Corollary 1 (size of the correlation gap).** *With f_κ, g_κ as in the
proof of Proposition 2, and Var(·), Cov(·, ·) taken with respect to
Lebesgue measure on \[0,1\],*

$$0\mspace{6mu} \leq \mspace{6mu} Q_{F}(\Pi) - \int_{0}^{1}\widetilde{Q}(\alpha;\Pi)\, d\alpha\mspace{6mu} = \mspace{6mu}\gamma\sum_{\kappa \in \Pi}^{}{Cov}\left( f_{\kappa},g_{\kappa} \right)\mspace{6mu} \leq \mspace{6mu}\gamma\sum_{\kappa \in \Pi}^{}\sqrt{Var\left( f_{\kappa} \right)}\,\sqrt{Var\left( g_{\kappa} \right)}\mspace{6mu} \leq \mspace{6mu}\frac{\gamma}{2}\sum_{\kappa \in \Pi}^{}\left\lbrack Var\left( f_{\kappa} \right) + Var\left( g_{\kappa} \right) \right\rbrack.$$

*Proof.* The identity is the proof of Proposition 2 read off: the linear
terms cancel exactly and the null terms differ by γ Σ_κ \[∫ f_κ g_κ − (∫
f_κ)(∫ g_κ)\] = γ Σ_κ Cov(f_κ, g_κ). The first inequality is the
Cauchy--Schwarz inequality for the covariance, the second the
arithmetic--geometric mean inequality. ∎

The corollary makes the gap computable from the graph and the partition.
What the gap means is a separate question, and the answer corrects the
earlier version of this paper, which read it as the cost of binarising.

**Remark 3 (the gap is not the cost of binarising).** Let every edge
carry the same membership c ∈ (0,1). Q_F is invariant to rescaling μ_E
by a constant, so Q_F(Π) equals the crisp modularity of the support
graph for every Π, and thresholding at any α ≤ c returns that support
graph: binarising is lossless. Yet f_κ(α) = (a_κ/c)·1\[α ≤ c\] with a_κ
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
returns, and Section 9.1 answers it with ground truth.

The normalisation in (9) is by the fuzzy mass W rather than by \|E_α\|,
and that choice is forced by the proof rather than by convenience. It
also means that (9) is not the modularity of the crisp graph G_α:
writing r = \|E_α\|/W, the internal term of (9) is the crisp one
weighted by r and the null term by r², so Q̃(α; Π) tends to zero as α →
1 simply because edges disappear, whereas the crisp modularity of G_α
need not. With global
normalisation both terms of Q_F decompose over the same Lebesgue measure
on α, which is what makes the identity in the proof exact; the linear
term is then the α-average of the per-cut internal fraction weighted by
\|E_α\|/W, so high cuts, which retain few edges and little mass,
contribute in proportion to the mass they carry. Under the more familiar
per-cut normalisation the null term divides a product of strength
profiles by a quantity that itself varies with α, the layer-cake
decomposition no longer commutes with that division, and no comparably
clean relation holds. Proposition 2 compares two evaluations of the *same* partition. The
shortcut a practitioner takes is to run a crisp detector on one level
set, with a resolution chosen for that level set, and keep whatever
partition it returns; whether that partition is worse for the fuzzy
graph, and whether it recovers the planted communities less well, are
empirical questions. Section 9.1 defines the cost operationally as

$$\text{loss}(\alpha)\mspace{6mu} = \mspace{6mu} Q_{F}\left( \Pi_{\text{fuzzy}} \right) - \max_{\gamma\prime}Q_{F}\left( \Pi_{\alpha}(\gamma\prime) \right),$$

the resolution γ′ being refitted for each level set, together with the
agreement between Π_α and Π_fuzzy and, where a planted partition
exists, the recovery of the truth by each.

## 6. Fuzzification of a point estimate with epistemic uncertainty

### 6.1 The trapezoidal partition

The estimator described in Section 8.2 returns, for each learner and KC,
a mean p̄ and a standard deviation σ over Monte-Carlo dropout passes
\[26\]; a forgetting transition then maps p̄ to p̄\^\*. Fix cut-points 0
\< α_unk \< α_par \< α_mas \< 1 and a widening factor η ≥ 0, and write
the interval bounds

$$lo = {\bar{p}}^{\,*} - \eta\sigma,\quad\quad hi = {\bar{p}}^{\,*} + \eta\sigma.\quad\quad\text{(10)}$$

The interval in (10) is not an ad hoc widening of the point estimate; it
is a level set of the dominating possibility distribution that the pair
(p̄\^\*, σ) determines, in the sense of the probability--possibility
transformations of \[27\].

**Lemma 2 (the interval is a level set of the Chebyshev possibility
envelope).** *Let P(p̄\^\*, σ) be the family of probability measures on
the real line with mean p̄\^\* and standard deviation σ, and define the
possibility distribution*

$$\pi_{C}(x)\mspace{6mu} = \mspace{6mu}\min\left\{ 1,\mspace{6mu}\frac{\sigma^{2}}{\left( x - {\bar{p}}^{\,*} \right)^{2}} \right\}.$$

*Then: (i) the possibility measure induced by π_C dominates every member
of the family, P(A) ≤ sup\_{x ∈ A} π_C(x) for every P ∈ P(p̄\^\*, σ) and
every measurable A; (ii) no symmetric possibility distribution with
strictly smaller cuts does, since for each d ≥ σ a two-point member of
the family attains equality; and (iii) for η ≥ 1 the α-cut of π_C at α =
η\^{−2} is exactly \[p̄\^\* − ησ, p̄\^\* + ησ\], so the interval (10) is
the η\^{−2}-cut of π_C, and the default η = 1 selects its core.*

*Proof.* (i) For measurable A put d = inf\_{x ∈ A} \|x − p̄\^\*\|. If d =
0 the bound is 1 and trivial; otherwise Chebyshev's probability
inequality gives P(A) ≤ P(\|X − p̄\^\*\| ≥ d) ≤ σ²/d² = sup\_{x ∈ A}
π_C(x). (ii) The distribution placing mass σ²/2d² at each of p̄\^\* ± d
and the rest at p̄\^\* lies in the family and satisfies P(\|X − p̄\^\*\| ≥
d) = σ²/d², so any dominating symmetric π must have π(p̄\^\* ± d) ≥ σ²/d²
= π_C(p̄\^\* ± d). (iii) π_C(x) ≥ η\^{−2} holds iff (x − p̄\^\*)² ≤ η²σ².
∎

The widening factor η therefore has a semantics rather than a tuning
role: it selects the possibility level of the cut, and the Chebyshev
inequality used here is the probability inequality of that name,
unrelated to the integral inequality in Proposition 2 beyond the
attribution. The lemma is the distribution-free instance of the
transformation family in \[27\], appropriate because Monte-Carlo dropout
supplies a mean and a dispersion but no distributional shape.

The three memberships are

$${\widetilde{\mu}}^{unk} = clip\left\lbrack \frac{\alpha_{par} - lo}{\alpha_{par} - \alpha_{unk}},\, 0,\, 1 \right\rbrack,\quad{\widetilde{\mu}}^{mas} = clip\left\lbrack \frac{hi - \alpha_{par}}{\alpha_{mas} - \alpha_{par}},\, 0,\, 1 \right\rbrack,\quad{\widetilde{\mu}}^{par} = \max\left( 0,\, 1 - {\widetilde{\mu}}^{unk} - {\widetilde{\mu}}^{mas} \right),\quad\quad\text{(11)}$$

with clip\[x,0,1\] = max(0, min(1,x)), followed by normalisation onto
Δ_3,

$$\varphi(c)\mspace{6mu} = \mspace{6mu}\frac{\left( {\widetilde{\mu}}^{unk},\,{\widetilde{\mu}}^{par},\,{\widetilde{\mu}}^{mas} \right)}{\max\left( \varepsilon,\mspace{6mu}{\widetilde{\mu}}^{unk} + {\widetilde{\mu}}^{par} + {\widetilde{\mu}}^{mas} \right)}.\quad\quad\text{(12)}$$

At η = 0 this is a trapezoidal fuzzy partition of the point-estimate
axis: μ_unk = 1 for p̄\^\* ≤ α_unk, μ_mas = 1 for p̄\^\* ≥ α_mas, μ_par
peaks at 1 exactly at p̄\^\* = α_par, and the transitions are linear. The
triple sums to one by construction and is monotone in p̄\^\* --- μ_mas
non-decreasing, μ_unk non-increasing. Section 9.4 verifies all three
properties over a dense grid and under 125 perturbations of the
cut-points.

The construction is *defensive* in the sense that μ_unk is evaluated at
the lower bound lo and μ_mas at the upper bound hi, so that widening the
interval can only increase the two extreme memberships at the expense of
μ_par. The system never asserts mastery it cannot warrant. Section 6.3
shows that this property, read as a design virtue, conceals a failure.

### 6.2 The footprint of uncertainty

The pair (lo, hi) is an interval-valued input. Pushing it through (11)
induces, for each label ℓ ∈ {unk, par, mas} with membership function m_ℓ
on \[0,1\], the interval

$${\underline{\mu}}^{\ell} = \min_{x \in \lbrack lo,\, hi\rbrack}m_{\ell}(x),\quad\quad{\overline{\mu}}^{\ell} = \max_{x \in \lbrack lo,\, hi\rbrack}m_{\ell}(x),\quad\quad\text{(13)}$$

which is a footprint of uncertainty in the sense of Section 3.4. The
extrema are available in closed form and require no iteration, and the
argument uses only shape properties, not the trapezoidal formulas
themselves: whenever m_unk is non-increasing, m_mas non-decreasing and
m_par unimodal on \[0,1\], each shoulder label attains its bounds at the
endpoints of \[lo,hi\], and m_par attains its maximum at m_par(α_par)
when α_par ∈ \[lo,hi\] and at the larger endpoint value otherwise, its
minimum at the smaller endpoint value. The construction therefore
applies to any Ruspini partition with monotone shoulders and a unimodal
core; the trapezoidal partition of Section 6.1 is the instance used
throughout. The footprint collapses to a point exactly when ησ = 0.
Figure 1 displays the three footprints and the type-1 collapse against
them.

![Fig. 1. Footprint of uncertainty of the three linguistic labels
induced by the interval \[p̄\* − ησ, p̄\* + ησ\] at half-width ησ = 0.10,
cut-points (α_unk, α_par, α_mas) = (0.35, 0.60, 0.80). Shaded band:
lower and upper memberships of Equation (13). Dashed curve: the type-1
collapse of Equations (11)--(12). For the two shoulder labels the
collapse stays inside its footprint; for the middle label it leaves the
footprint on part of the axis, the phenomenon quantified over the full
grid in Section 9.5.](./media/rId44.png){width="5.833333333333333in"
height="1.8333333333333333in"}

**Fig. 1.** Footprint of uncertainty of the three linguistic labels
induced by the interval \[p̄\* − ησ, p̄\* + ησ\] at half-width ησ = 0.10,
cut-points (α_unk, α_par, α_mas) = (0.35, 0.60, 0.80). Shaded band:
lower and upper memberships of Equation (13). Dashed curve: the type-1
collapse of Equations (11)--(12). For the two shoulder labels the
collapse stays inside its footprint; for the middle label it leaves the
footprint on part of the axis, the phenomenon quantified over the full
grid in Section 9.5.

**Remark 3 (why the footprint has this shape).** The secondary grade is
interval-valued rather than Gaussian because the estimator supplies
exactly an interval: Monte-Carlo dropout returns a mean and a
dispersion, and committing to a shape over \[lo, hi\] would assert
information the R passes do not provide. The primary memberships are
piecewise linear because they are derived, not designed: given the
type-1 partition of Section 6.1 and the interval input, Equation (13) is
the exact image, so the only genuine design choice is the partition
itself. The trapezoidal partition is the member of the admissible class
on which the cut-points carry their linguistic semantics linearly and
the Ruspini sum-to-one property \[37\] holds by construction; a Gaussian
primary partition would keep the closed-form extrema, since a Gaussian
shoulder is still monotone on each side of its mode, but would lose the
sum-to-one reading of the triple, so the price would be semantic rather
than computational. The same information constraint settles interval
type-2 over general type-2: a general type-2 set needs a secondary
membership over each primary interval, and by Lemma 2 a mean and a
dispersion determine the interval as a level set of the dominating
envelope and nothing more, so any non-constant secondary grade would
encode an assumption the dropout passes do not license. Reading \[lo,
hi\] as an interval-valued input is the minimal-commitment
representation the probability--possibility consistency principle
\[38,27\] licenses. Carrying (13) forward makes the
zone-of-proximal-development gate of (P1) three-valued rather than
two-valued: a candidate *passes* if μ̲\^{par} ≥ τ_ZPD, *fails* if
μ̄\^{par} \< τ_ZPD and is otherwise *indeterminate*, meaning that the
graph and the estimator together do not determine whether the KC lies in
the learner's zone of proximal development. Section 9.5 measures the
indeterminate region on a grid.

### 6.3 The cautious collapse, and why its defect is its own

Equations (11)--(12) are the construction our earlier pipelines used;
we know of no published pipeline that uses exactly this collapse, and
the earlier version of this paper was wrong to call it common practice.
They select, for each shoulder label, the endpoint at which that
label's membership is largest, and give μ_par the remainder. This is a
type reduction by a fixed worst-case-toward-caution criterion rather
than by Karnik--Mendel iteration \[22\]. Whether the result is an
admissible embedded type-1 set of the footprint it comes from is a
separate question, and the answer is no, for two reasons; the second is
the one that matters for a decision.

*Normalisation destroys embeddedness.* The renormalisation (12) is
applied *after* the endpoint selection. Three quantities selected at
three different points of the interval are divided by their own sum, and
the renormalised μ_par need not lie between its own footprint bounds.
Section 9.5 measures the failure rate on a grid. Embeddedness is a
criterion internal to the type-2 representation, and we make no claim
that violating it harms a decision; it is recorded because it is a
cheap diagnostic.

*The collapse degenerates under moderate uncertainty, and the threshold
is structural.* Widening the interval can only raise μ_unk and μ_mas,
hence can only lower μ_par, which is the quantity the gate tests. The
degeneration point admits a closed form, so it is a property of every
trapezoidal collapse and not an artefact of the particular cut-points.
Write a = α_par − α_unk and b = α_mas − α_par and h = ησ. On the region
where neither shoulder clips, the collapsed sum μ̃\^{unk} + μ̃\^{mas}
changes at rate 1/b − 1/a in the point estimate, so the collapsed μ_par
is maximised at the endpoint p̄\^\* = α_par − h when a ≥ b (at α_par + h
otherwise), where one shoulder vanishes and the other equals 2h divided
by the wider band. Hence

$$\max_{{\bar{p}}^{\,*}}\mspace{6mu}\mu_{collapse}^{par}\mspace{6mu} = \mspace{6mu} 1 - \frac{2h}{\max(a,b)},$$

exactly (the renormalisation of (12) is inactive there, the unnormalised
triple already summing to one), and the gate at threshold τ_ZPD rejects
every candidate exactly when h exceeds

$$h^{*}\mspace{6mu} = \mspace{6mu}\frac{1}{2}\,\left( 1 - \tau_{ZPD} \right)\,\max(a,b).$$

For the default cut-points and τ_ZPD = 0.4 this gives h\^\* = 0.075,
which the measured grid brackets (Table 13). The mechanism is a
three-line consequence of the construction: widening the interval can
only raise the two shoulder memberships, hence only lower μ_par. It is
therefore a defect of the cautious collapse and not of type-1
representations in general, and the comparison that matters is with
gates that do not share the design. Section 9.5 compares four: the gate
on the point estimate p̄ alone; the type-1 gate on the interval image,
pass iff min_{x ∈ \[lo, hi\]} m_par(x) ≥ τ_ZPD, which is the "pass"
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
that level is P(\|X − p̄\| \< ησ) ≥ 1 − η\^{−2} = 0, so the choice has
no coverage content; η = 2 and η = 3 give 0.75 and 0.89, and Table 1
now lists η as a label-semantics choice. And with R = 8 dropout passes
the relative standard error of σ is about 27%, while dropout dispersions
are known to be poorly calibrated under shift \[50\], so Lemma 2's
semantics presuppose a σ that Section 9 never checks against held-out
outcomes. For these reasons the interval type-2 state is kept in the
released code as a representation and its gate comparison on a trained,
evaluated estimator is deferred to the outcome study of Section 10.2;
the footprint and Nie--Tan reduction \[23\] cost a few hundredths of a
millisecond per state and are not a consideration.

## 7. Path composition and the limits of t-norms

### 7.1 No t-norm compensates

Equation (1) composes edge memberships along a path by the minimum. A
natural design question is whether some other t-norm, in particular a
member of a parametric family, would allow a strong prerequisite edge to
partially offset a weak one. It would not, and the reason is Equation
(2): the minimum is the largest t-norm, so max-min composition is the
upper envelope of the entire class.

**Fact 1.** *Let T be any t-norm and let S_T(π) = T\_{(u,v) ∈ π}
μ_E(u,v) denote the composition of the memberships along a path π. Then
S_T(π) ≤ min\_{(u,v) ∈ π} μ_E(u,v) for every π, and consequently max_π
S_T(π) ≤ μ_E\^∞(a,b) for all a, b.*

*Proof.* Immediate from (2) by induction on the length of π, using
associativity; the second claim follows by taking the maximum over Π_ab
of both sides. ∎

Fact 1 is elementary --- the bound (2) is a standard consequence of the
t-norm axioms, established in the classical literature \[8\] --- and it
is recorded as a fact rather than claimed as a contribution. It also
draws the boundary of the operator class this paper considers.
Aggregation families that do permit compensation exist and are well
studied: uninorms \[39\] behave conjunctively below a neutral element e
∈ (0,1) and disjunctively above it, so a chain of strong links can
aggregate to a value exceeding its weakest member, and nullnorms
compensate dually around an absorbing element. These operators are
excluded here on semantic rather than technical grounds: a prerequisite
path can be no more passable than its least passable link, so the
modelling requirement is exactly the bound (2), and an operator designed
to violate it above its neutral element answers a different question.
The compensatory rows of Table 11 are included to make the cost of
abandoning that requirement visible, not to propose them. Its role is to
frame the empirical question: the design choice a parametric t-norm
family exposes is *how much path strength to erode below the weakest
link*, never whether to compensate for it, so what remains open, and
what has not to our knowledge been measured, is the size of that erosion
and its effect on the induced ranking on real fuzzy graphs. Section 9.3
supplies both measurements. Compensation in the intended sense requires
leaving the class for an averaging operator or a uninorm; the OWA
operators of Section 3.3 are the natural candidates, and Section 9.3
includes three of them for contrast, where their defining behaviour ---
sitting *above* the weakest link --- is exactly what no t-norm can do.

### 7.2 Containment of the operators in use

The three operators most often compared in practice are the minimum, the
product and the Łukasiewicz t-norm. They are not independent of the two
parametric families but members or limits of them: T_1\^Y = T_L and
T_p\^Y → min as p → ∞; T_p\^{SS} → ab as p → 0, T_1\^{SS} = T_L and
T_p\^{SS} → min as p → −∞. We verified these identities numerically over
a 100 × 100 grid on (0,1\]², with maximum deviation 1.1 × 10\^{−16} for
the Yager--Łukasiewicz identity and 0 in the remaining cases. A
comparison restricted to the three standard operators therefore spans
both families at their endpoints; what a sweep adds is the interior, and
Section 9.3 reports it on real graphs. Terminology: the product and the
Schweizer--Sklar norms with p \< 0 are strict (continuous and strictly
increasing on (0,1\]²); Łukasiewicz, the Yager family and
Schweizer--Sklar with p \> 0 are nilpotent. All are continuous
Archimedean, and "non-idempotent" is the common property that separates
every one of them from the minimum.

### 7.3 Best paths under Archimedean t-norms are shortest paths; only max-min is ordinally invariant

Every continuous Archimedean t-norm T has an additive generator g :
\[0,1\] → \[0, ∞\], continuous and strictly decreasing with g(1) = 0,
such that T(a, b) = g\^{(−1)}(g(a) + g(b)), where g\^{(−1)}(y) =
g\^{−1}(min(y, g(0))) is the pseudo-inverse \[8\]. By associativity
the composition along a path π is S_T(π) = g\^{(−1)}(Σ\_{e ∈ π}
g(μ_E(e))), and since g\^{(−1)} is non-increasing the best path from a
to b under T is a path of minimum additive length Σ g(μ_E(e)) with
non-negative edge lengths: an ordinary shortest-path problem, solved
exactly by Dijkstra's algorithm in O(\|E\| log \|C\|), and with a hop
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

  Yager, p \> 0           (1 − x)\^p            1           nilpotent

  Schweizer--Sklar, p \< 0 (1 − x\^p)/p          ∞           strict

  Schweizer--Sklar, p \> 0 (1 − x\^p)/p          1/p         nilpotent

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

**Proposition 3 (ordinal invariance).** *Let φ : \[0,1\] → \[0,1\] be
strictly increasing and let G_F\^φ carry memberships φ ∘ μ_E. (i) For
every pair (a, b) the sets of max-min best paths on G_F and G_F\^φ
coincide, and the ordering of targets by μ_E\^∞(a, ·) is the same on
both graphs. (ii) For every continuous Archimedean t-norm T there exist
a graph and a strictly increasing φ for which the T-best path from a to
b changes.*

*Proof.* (i) S_min(π) = min_{e ∈ π} μ_E(e), and φ(min_e μ_E(e)) = min_e
φ(μ_E(e)) because φ is increasing, so S_min\^φ(π) = φ(S_min(π)) for
every path, and φ preserves the order of path strengths and of target
strengths. (ii) Take a two-hop path π_1 with memberships (x, x) and a
one-hop path π_2 with membership y, x \> y. Under T, π_1 beats π_2 iff
2g(x) \< g(y). Since g is continuous with g(1) = 0 and g(0) \> 0,
choose x close to 1 so that 2g(x) \< g(y); then choose φ increasing
with φ(x) = x′ and φ(y) = y′ such that 2g(x′) \> g(y′), which is
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
sweep of Section 9.3 should be read with that in mind.

## 8. Construction pipeline

This section describes the pipeline used to obtain fuzzy knowledge
graphs from interaction logs and the graded states that sit on them. It
is the instantiation on which Sections 5 to 7 are exercised, and the
released script reproduces it end to end.

### 8.1 Constructing the fuzzy knowledge graph

For every ordered pair (c_a, c_b), c_a ≠ c_b, three signals in \[0,1\]
are computed from the learner corpus. *Temporal precedence* is the
normalised rate at which c_a is practised before c_b within the same
learner's history,

$$prec\left( c_{a},c_{b} \right)\mspace{6mu} = \mspace{6mu}\frac{\left| \{(t,t\prime):t < t\prime,\ c_{k_{t}} = c_{a},\ c_{k_{t\prime}} = c_{b}\} \right|}{\left| \{(t,t\prime):t \neq t\prime,\ c_{k_{t}} = c_{a}\  \vee \ c_{k_{t\prime}} = c_{b}\} \right|}.\quad\quad\text{(14)}$$

*Empirical mastery gain* is the difference in mean correctness on c_b
conditional on prior partial mastery of c_a, mapped into \[0,1\] by a
logistic squash. An optional *taxonomy prior* contributes 0.9 when an
expert curriculum places c_b downstream of c_a and 0 otherwise.

The three signals are aggregated by an OWA operator with w = (0.5, 0.3,
0.2):

$$\mu_{E}\left( c_{a},c_{b} \right)\mspace{6mu} = \mspace{6mu}\sum_{j = 1}^{3}w_{j}\, b_{j},\quad\quad b_{1} \geq b_{2} \geq b_{3}.\quad\quad\text{(15)}$$

OWA is used rather than a fixed linear combination because it is
order-based: an edge is rewarded for the presence of any strong signal
without a single weak signal vetoing it. The weight vector is fixed by a
target orness of 0.65 in the sense of Section 3.3, a moderately
optimistic aggregation appropriate to pooling heterogeneous evidence;
for a new corpus the weights are regenerated from the same orness target
through Yager's regular-increasing-monotone quantifier, so that the
semantics rather than the numbers transfer. Edges with μ_E \< τ_E = 0.2
are pruned, which is what makes the hypothesis μ_E : E → (0,1\] of Lemma
1 and Proposition 2 hold by construction.

Fuzzy Leiden is then run at two resolutions to produce a two-tier
partition; each community is the unit that a downstream recommender
summarises and indexes, which is why the partition is kept crisp (Remark
2).

### 8.2 Estimating the graded state

A two-layer self-attention encoder with monotonic decay, in the style of
\[40,41\], produces a point-valued mastery estimate per KC. Dropout is
kept active at inference and R = 8 independent forward passes give the
sample mean p̄ and standard deviation σ, which under the interpretation
of \[26\] approximate the posterior predictive mean and standard
deviation; in the taxonomy of \[42\] the dispersion estimates the
reducible, epistemic component of the predictive uncertainty, which is
what Section 6 propagates. An exponential forgetting kernel with per-KC
rate f_c, in the style of \[43\], transports p̄ from the last interaction
time to query time. The pair (p̄\^\*, σ) is then fuzzified by Section 6.

### 8.3 The gate and the path score

The two fuzzy-set constraints of Section 4.3 are consumed as follows. A
candidate next concept passes the zone-of-proximal-development gate (P1)
when μ\^{par}(c) ≥ τ_ZPD and μ\^{mas}(c) \< α_mas; under the interval
type-2 state of Section 6.2 the gate is three-valued, evaluated on the
footprint bounds themselves and not on any type-reduced point, and
indeterminate candidates are routed to an abstention outcome rather than
silently dropped. A path π from the anchor to a target is scored by

$$Score(\pi)\mspace{6mu} = \mspace{6mu}\min_{(u,v) \in \pi}\mu_{E}(u,v)\mspace{6mu} \cdot \mspace{6mu}\prod_{c_{j} \in interior(\pi)}^{}\mathbf{1}\left\lbrack \mu^{par}\left( c_{j} \right) \geq \tau_{ZPD} \right\rbrack,\quad\quad\text{(16)}$$

so that the weakest link drives the score, in the max-min semantics of
Equation (1), and any interior node outside the learner's zone of
proximal development sends the score to zero. The constraint (P2) is
enforced by requiring μ_E ≥ τ_path on every hop. Equation (16) is the
only place where the three shortcuts of this paper meet: its minimum is
the composition operator of Section 7, its indicator consumes the state
of Section 6, and the candidate pool it ranges over is a community of
the partition of Section 5. The retrieval and generation layer that sits
on top of (16) in a deployed recommender is outside the scope of this
paper (Section 10.2).

### 8.4 Parameters

Table 1 lists the parameters that the results of Section 9 depend on,
their defaults and the source of each value. Three sources are
distinguished: theory (T), linguistic-label semantics (P) and a
data-driven fit on a validation split (D). Section 9.4 measures the
sensitivity of the fuzzification to its cut-points and of the
construction to the OWA orness, and Section 9.1 reports the case in
which the resolution defaults fail to transfer between graphs of
different density.

**Table 1.** Parameters, defaults, and source of each value (T = theory,
P = label semantics, D = data-driven fit).

  --------------------------------------------------------------------------
  Symbol         Meaning        Default        Where          Source
  -------------- -------------- -------------- -------------- --------------
  w = (w₁,w₂,w₃) OWA weights    (0.5, 0.3,     Eq. (15)       T (orness
                                0.2)                          0.65)

  τ_E            Edge-prune     0.2            §8.1           P
                 threshold                                    

  γ₁, γ₂         Leiden         0.5, 1.0       Eq. (7)        T (two-tier);
                 resolutions                                  refit per
                                                              corpus by
                                                              Algorithm 1

  R              MC-dropout     8              §8.2           P
                 passes                                       

  η              Uncertainty    1.0            Eq. (10)       P (core of
                 widening                                     π_C; no
                                                              coverage)

  α_unk, α_par,  Fuzzy          0.35, 0.60,    Eq. (11)       P / D
  α_mas          cut-points     0.80                          

  τ_ZPD          ZPD threshold  0.4            Eq. (16)       P / D

  τ_path         Minimum path   0.3            §8.3           P
                 edge weight                                  
  --------------------------------------------------------------------------

The cut-points (α_unk, α_par, α_mas) = (0.35, 0.60, 0.80) follow the
three-band mastery semantics standard in fuzzy student modelling
\[31,25\]: α_par is the competence threshold and α_mas the mastery
threshold. Where expert labels are available they are fitted by
maximising Cohen's κ between the induced band and the expert-assigned
level. The pruning threshold τ_E, the number of dropout passes R and the
path-edge floor τ_path were set by hand in earlier work and carried
over; no validation split was used to choose them, and Table 1 now
records them as practitioner defaults (P) rather than as fitted
values. The orness sweep of Section 9.4 shows that τ_E is insensitive
to the aggregation; the sensitivity of the results to R and τ_path has
not been measured.

## 9. Experiments

All computations were performed on a single CPU core. Two corpora supply
real fuzzy graphs. **MOOCCubeX** \[3\], restricted to its
Computer-Science discipline, ships an expert-curated
concept-prerequisite graph with continuous \[0,1\] confidences rather
than a binary flag, so μ_E is available directly from the release.
Intersecting the 725 ground-truth prerequisite pairs in
`prerequisites/cs.json` with the 1857 assessed concepts in
`relations/concept-problem.txt`, and keeping edges whose endpoints both
carry at least one graded problem, gives 59 KCs, 91 directed
prerequisite edges and 119 problems, with μ_E ranging over 0.400 / 0.934
/ 0.999 (min / mean / max). The behavioural interaction log of MOOCCubeX
(22.4 GB) was not downloaded; a streaming script that computes the
corresponding statistics from the public log is released with the code.

Three public **ASSISTments** releases (2009, 2015, 2017) supply
interaction logs from which Section 8.1 constructs fuzzy graphs end to
end, at two granularities: the KC level, where nodes are knowledge
components (skills in 2009 and 2017, skill-builder problem sets in
2015), and the item level, where nodes are individual problems. The
preprocessing is stated in full and shipped with the code: rows with
missing skill are dropped, scaffolding rows are removed where the
release marks them (original = 1 in 2009 and 2017), duplicate log
entries are removed, responses outside {0,1} are dropped in 2015,
sequences are ordered by the release's log order, and learners with
fewer than 15 retained interactions are excluded. The precedence signal
is the directional share of within-horizon co-occurrences, the
mastery-gain signal uses a logistic squash with slope 5, and the
remaining construction parameters are those of Table 1. ASSISTments 2012
was not available in the same preprocessed form.

Tables 2, 3, 4, 5, 8, 10, 11 and 12 are computed from real public data;
the six real graphs derive from four corpora (two granularities each of
ASSISTments 2009 and 2017, one 2015 graph, one MOOCCubeX graph). Tables
6, 7 and 9 are computed from planted and synthetic graphs, and Tables 13
and 14 from dense parameter grids generated by the released code; all
are labelled as such.

### 9.1 Fuzzy-graph modularity on real graphs

**MOOCCubeX-CS.** Fuzzy Leiden on the 59-node graph finds 4 communities
at γ₁ = 0.5 with Q_F = 0.672 and sizes {21,20,16,2}, and 6 communities
at γ₂ = 1.0 with Q_F = 0.524 and sizes {19,14,10,8,6,2}. Every one of
the 10 communities induces a weakly connected subgraph of the support
graph, which is the sanity check for Lemma 1(i). The finer resolution splits the three large clusters
while a 21-node cluster at the coarse scale survives largely intact as
the 19-node cluster at the fine one.

**Table 2.** Fuzzy Leiden on the real MOOCCubeX-CS prerequisite graph
(best of 10 seeds).

  -----------------------------------------------------------------------
  Quantity                Coarse (γ = 0.5)        Fine (γ = 1.0)
  ----------------------- ----------------------- -----------------------
  Communities found       4                       6

  Fuzzy-graph modularity  0.672                   0.524
  Q_F                                             

  Largest community size  21                      19

  Smallest community size 2                       2

  Communities             0 / 4                   0 / 6
  disconnected (Lemma                             
  1(i))                                           
  -----------------------------------------------------------------------

**ASSISTments.** Section 8.1 was run end to end on the three releases
with a precedence horizon of 10 interactions, a minimum support of 20
co-occurrences, OWA weights w = (0.5, 0.3, 0.2), no taxonomy prior and
pruning at τ_E = 0.2, under the preprocessing stated above. Table 3
reports the resulting graphs and Table 4 the partitions at four
resolutions, each the best of five seeds.

**Table 3.** Fuzzy knowledge graphs constructed from real ASSISTments
interaction logs by the released pipeline.

  ---------------------------------------------------------------------------
  Corpus    Learners   Interactions   Nodes     Fuzzy     Mean μ_E  Build (s)
                                                edges               
  --------- ---------- -------------- --------- --------- --------- ---------
  2009, KC  2,445      247,625        110       5,059     0.467     4.5
  level                                                             

  2009,     2,445      247,625        371       4,950     0.421     9.9
  item                                                              
  level                                                             

  2015, KC  11,996     629,709        100       6,939     0.443     10.7
  level                                                             

  2017, KC  1,686      248,886        83        5,493     0.440     4.1
  level                                                             

  2017,     1,686      248,886        867       33,575    0.452     5.3
  item                                                              
  level                                                             
  ---------------------------------------------------------------------------

**Table 4.** Fuzzy Leiden on the graphs of Table 3, reported as Q_F /
number of communities at four resolutions. "Conn." is the Lemma 1(i)
check on every community at every resolution; "Mono." is the Lemma 1(iv)
check over ten successive optimisation iterations at every resolution.

  ----------------------------------------------------------------------------
  Corpus     γ = 0.5    γ = 1.0    γ = 2.0    γ = 5.0    Conn.      Mono.
  ---------- ---------- ---------- ---------- ---------- ---------- ----------
  2009, KC   0.500 / 2  0.099 / 6  0.004 / 50 −0.046 /   pass       pass
  level                                       81                    

  2009, item 0.938 / 28 0.876 / 28 0.751 / 28 0.378 / 28 pass       pass
  level                                                             

  2015, KC   0.500 / 1  0.059 / 3  −0.015 /   −0.052 /   pass       pass
  level                            75         93                    

  2017, KC   0.500 / 1  0.026 / 3  −0.024 /   −0.064 /   pass       pass
  level                            73         78                    

  2017, item 0.723 / 16 0.564 / 20 0.361 / 24 0.181 / 86 pass       pass
  level                                                             
  ----------------------------------------------------------------------------

Four findings follow from Tables 2 to 4, of which the third is negative;
Tables 5 to 8 then report the resolution procedure, the null-model test and the binarisation shortcut.

*The inherited properties were not violated.* Across five graphs and
four resolutions, twenty partitions in total, every returned community
induces a weakly connected subgraph of the support graph, on graphs up
to 867 nodes and 33575 fuzzy edges, and over ten successive
optimisation iterations per configuration Q_F never decreased. These
are sanity checks on the implementation of a proved lemma, not
evidence for it; boundedness is visible in the table as well, the
negative entries at γ = 5.0 lying above −γ as Lemma 1(ii) requires.

*Construction is inexpensive.* Precedence and mastery-gain estimation
over six hundred thousand interactions, followed by OWA aggregation and
pruning, completes in 10.7 s on one core for the largest corpus.
Construction is an offline, cached cost.

*The resolution defaults do not transfer.* At the KC level the defaults
γ₁ = 0.5, γ₂ = 1.0, chosen on the sparse MOOCCubeX-CS curriculum graph,
produce a degenerate partition: γ = 0.5 returns the trivial single
community on the 2015 and 2017 KC-level graphs and a two-community split
of identical value on the 2009 one, and Q_F = 0.500 in those rows is the
trivial value 1 − γ rather than evidence of structure. The cause is
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
a smaller θ. A KC-level ASSISTments graph has of the order of
100 nodes and 5000--7000 retained edges, a mean total degree near 100,
whereas MOOCCubeX-CS has 59 nodes and 91 edges, a mean degree near 3.
The resolution therefore belongs with the parameters that must be
refitted per corpus rather than transferred, following Lemma 1(i) and
Algorithm 1, and Table 1 records this.

*Granularity matters more than corpus.* The item-level graphs return
partitions with Q_F between 0.18 and 0.94 across the resolution grid,
while the KC-level graphs behave like dense co-practice graphs, which
is what they are. The 28 communities of the 2009
item-level graph at every resolution are its 28 weakly connected
components: the released code reports the component structure of
every graph (MOOCCubeX-CS 2 components; 2009 KC level 2, of sizes 108
and 2; 2009 item level 28, the largest of 44 nodes; 2015 and 2017 KC
level 1; 2017 item level 7, the largest of 849 nodes), and the
Fuzzy-Leiden partition of the 2009 item-level graph coincides with the
components (NMI 1.00), as does the two-community split of the 2009
KC-level graph at γ = 0.5. Items of one skill-builder problem set
co-occur and items of different sets do not, so Q_F ≈ 0.94 on that
graph reflects the construction, and Fuzzy Leiden recovers components.
The 2017 item-level graph is different: 849 of its 867 nodes lie in
one component and the 16-community partition at γ = 0.5 has NMI 0.16
with the components, so there the partition is not a component
labelling. The co-practice reading of the KC-level graphs is the
following: with a hundred skills and hundreds of thousands of interactions,
almost every pair of skills co-occurs within the precedence window, so
precedence says something about ordering and little about whether a
relation exists. The prerequisite reading of μ_E is well founded at the
item level and questionable at the KC level for this corpus family, a
scope condition on Equation (15), not on Section 5.

*Algorithm 1 seeds the resolution and, with the null-model test,
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
structure detected" (ΔQ = 0.001). On the six real graphs the test
separates the two granularities cleanly. The item-level graphs carry
structure the null model cannot produce: ΔQ = 0.71 (2009) and 0.48
(2017) at γ = 1. The three KC-level graphs do not: their excess over
the null is 0.056, 0.031 and 0.008, statistically separated (the null
standard deviation is 10\^{−3}) but below any useful floor, so the
outcome is "no structure detected" and the 74- to 87-community
partitions of Table 5 are not communities. The surprise is
MOOCCubeX-CS. Its expert-curated graph has Q_F = 0.524 at γ = 1, and a
degree-preserving null with the same memberships has 0.527 ± 0.018:
the modularity of this graph is what any 59-node directed graph with
91 edges and these degrees has, and Tables 2 and 10 measure the
optimiser on it, not community structure in it. The earlier version
treated that graph as the curated reference; it is a sparse near-tree
on which modularity is uninformative, which Section 9.2 already
identified as the unstable regime.

**Table 7.** The revised Algorithm 1 on planted and synthetic graphs
and on the six real graphs (ρ₁ = 1.25, ρ₂ = 2, five seeds; null model:
20 degree-preserving rewirings with memberships permuted, three seeds
each, at γ_ref = 1; z_min = 3, ΔQ_min = 0.10). Q_F(1) and null(1) are
the best observed and the mean null Q_F at γ = 1. The real-graph rows
were produced by the repository pipeline (see the note in Data
availability).

  ----------------------------------------------------------------------------------------------------------------
  Graph                      γ₁ accepted  Communities  Q_F at γ₁  Q_F(1)    null(1)   ΔQ      z        Outcome
  -------------------------- ------------ ------------ ---------- --------- --------- ------- -------- ---------------------
  Planted, p_in = 0.8        0.262        11           0.824      0.762     0.252     0.511   \> 10²   structure

  Planted, p_in = 1/12       108.842      582          -0.179     0.255     0.249     0.006   2.4      no structure detected

  Dense co-practice-like     4.543        109          -0.041     0.067     0.067     0.001   0.4      no structure detected

  MOOCCubeX-CS               9.404        29           0.058      0.524     0.527     -0.004  -0.2     no structure detected

  2009, KC level             4.642        81           -0.040     0.102     0.046     0.056   32.9     no structure detected

  2009, item level           0.223        28           0.972      0.873     0.159     0.714   \> 10²   structure

  2015, KC level             4.529        92           -0.047     0.059     0.028     0.031   31.0     no structure detected

  2017, KC level             2.401        77           -0.029     0.025     0.017     0.008   7.9      no structure detected

  2017, item level           0.619        16           0.676      0.564     0.085     0.479   \> 10²   structure
  ----------------------------------------------------------------------------------------------------------------


**Table 5.** Seeding steps of Algorithm 1 on the six real fuzzy graphs
(ρ₁ = 1.25, ρ₂ = 2, five seeds, θ_ARI = 0.8). c/W is the balanced spectral cut fraction of
step 2; γ₁ the accepted coarse resolution; Q_F and the community count
are for the best-of-seeds partition at γ₁.

  -------------------------------------------------------------------------------
  Graph           c/W     2c/W    Doublings   γ₁ accepted Q_F at γ₁ Communities
  --------------- ------- ------- ----------- ----------- --------- -------------
  MOOCCubeX-CS    0.153   0.305   0           0.382       0.718     4

  2009, KC level  0.346   0.691   2           3.456       −0.025    74

  2009, item      0.116   0.232   0           0.290       0.964     28
  level                                                             

  2015, KC level  0.342   0.684   2           3.420       −0.034    87

  2017, KC level  0.366   0.732   2           3.662       −0.046    77

  2017, item      0.265   0.531   1           1.326       0.475     20
  level                                                             
  -------------------------------------------------------------------------------

*Binarising before partitioning, measured against ground truth.*
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

  ----------------------------------------------------------------------------------------------------------------------------------------------
  Mean  SD    Shift  Fuzzy (NMI; gap)  α = 0.4           α = 0.6           α = 0.8           q = 0.25          q = 0.50          q = 0.75        
  ----- ----- ------ ----------------  ----------------  ----------------  ----------------  ----------------  ----------------  ----------------
  0.3   0.05  0.0    1.00; 0.101       0.60/0.590/0.04   ---               ---               1.00/0.001/0.80   0.97/0.012/0.57   0.68/0.181/0.30 

  0.3   0.05  0.5    1.00; 0.059       1.00/-0.000/0.75  0.61/0.815/0.00   ---               1.00/0.000/0.91   1.00/0.000/0.67   0.99/0.015/0.35 

  0.3   0.15  0.0    1.00; 0.070       0.68/0.151/0.42   0.60/0.548/0.08   0.61/0.629/0.00   0.99/0.000/0.90   0.97/0.007/0.70   0.68/0.152/0.42 

  0.3   0.15  0.5    1.00; 0.045       1.00/0.001/0.68   0.79/0.314/0.23   0.61/0.800/0.02   1.00/-0.000/0.95  1.00/0.000/0.74   0.97/0.024/0.43 

  0.3   0.25  0.0    0.98; 0.040       0.84/0.040/0.65   0.43/0.271/0.37   0.59/0.496/0.13   0.99/0.001/0.97   0.96/0.004/0.84   0.70/0.099/0.55 

  0.3   0.25  0.5    1.00; 0.026       0.99/0.002/0.77   0.91/0.051/0.50   0.57/0.544/0.21   1.00/0.000/0.99   1.00/0.001/0.84   0.92/0.042/0.53 

  0.5   0.05  0.0    1.00; 0.044       1.00/0.000/0.98   0.60/0.604/0.03   ---               1.00/0.001/0.78   0.97/0.012/0.54   0.64/0.222/0.28 

  0.5   0.05  0.5    1.00; 0.017       1.00/-0.000/0.87  1.00/-0.000/0.86  0.81/0.363/0.15   1.00/0.000/0.90   1.00/-0.000/0.65  0.99/0.015/0.34 

  0.5   0.15  0.0    1.00; 0.033       0.99/0.001/0.83   0.73/0.145/0.36   0.60/0.597/0.03   0.99/0.001/0.85   0.97/0.009/0.62   0.67/0.176/0.35 

  0.5   0.15  0.5    1.00; 0.012       1.00/0.000/0.90   1.00/0.000/0.78   1.00/0.004/0.44   1.00/0.000/0.92   1.00/0.000/0.70   0.99/0.015/0.38 

  0.5   0.25  0.0    1.00; 0.021       0.99/0.001/0.82   0.89/0.035/0.57   0.42/0.353/0.25   1.00/0.000/0.91   0.96/0.009/0.71   0.70/0.148/0.41 

  0.5   0.25  0.5    1.00; 0.007       1.00/0.000/0.91   1.00/0.000/0.80   1.00/0.002/0.61   1.00/0.000/0.96   1.00/0.000/0.75   0.98/0.020/0.41 

  0.7   0.05  0.0    1.00; 0.019       1.00/0.000/1.00   1.00/-0.000/0.98  0.60/0.611/0.02   1.00/0.001/0.77   0.96/0.014/0.53   0.68/0.207/0.27 

  0.7   0.05  0.5    1.00; 0.009       1.00/0.000/1.00   1.00/-0.000/0.81  1.00/-0.000/0.66  1.00/-0.000/0.82  1.00/0.000/0.58   0.99/0.013/0.30 

  0.7   0.15  0.0    1.00; 0.013       1.00/-0.000/0.98  1.00/0.001/0.82   0.76/0.121/0.35   1.00/0.001/0.82   0.97/0.011/0.59   0.68/0.181/0.31 

  0.7   0.15  0.5    1.00; 0.006       1.00/0.000/0.97   1.00/-0.000/0.85  1.00/0.000/0.62   1.00/0.000/0.85   1.00/0.000/0.61   0.98/0.017/0.32 

  0.7   0.25  0.0    1.00; 0.007       1.00/0.000/0.95   0.99/0.001/0.83   0.95/0.015/0.59   1.00/0.000/0.88   0.97/0.008/0.65   0.72/0.154/0.34 

  0.7   0.25  0.5    1.00; 0.003       1.00/0.000/0.95   1.00/0.000/0.87   1.00/0.000/0.74   1.00/0.000/0.89   1.00/0.000/0.65   0.99/0.014/0.33 

  0.9   0.05  0.0    1.00; 0.004       1.00/-0.000/1.00  1.00/-0.000/1.00  1.00/-0.000/0.97  1.00/0.001/0.77   0.96/0.016/0.52   0.70/0.181/0.27 

  0.9   0.05  0.5    1.00; 0.002       1.00/0.000/1.00   1.00/0.000/1.00   1.00/0.000/0.95   1.00/0.000/0.78   1.00/0.000/0.53   0.98/0.017/0.27 

  0.9   0.15  0.0    1.00; 0.001       1.00/0.000/0.99   1.00/0.000/0.97   1.00/0.000/0.88   1.00/0.000/0.81   0.97/0.012/0.55   0.71/0.171/0.28 

  0.9   0.15  0.5    1.00; 0.001       1.00/0.000/0.99   1.00/0.000/0.97   1.00/0.000/0.91   1.00/0.000/0.81   1.00/0.000/0.54   0.98/0.018/0.27 

  0.9   0.25  0.0    1.00; 0.000       1.00/0.000/0.99   1.00/0.000/0.98   1.00/0.000/0.95   1.00/0.001/0.83   0.97/0.013/0.55   0.66/0.209/0.28 

  0.9   0.25  0.5    1.00; 0.000       1.00/0.000/0.99   1.00/0.000/0.98   1.00/0.000/0.96   1.00/0.000/0.82   1.00/0.000/0.64   1.00/0.000/0.64 

  ----------------------------------------------------------------------------------------------------------------------------------------------


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

Table 8 repeats the shortcut on the six real graphs with the same
protocol: Fuzzy Leiden and crisp Leiden on each level set are scanned
over γ′ ∈ {0.25, 0.5, 1, 2, 4} and five seeds, the partition with the
best Q_F on the fuzzy graph is kept, and each cut reports the loss, the
NMI between Π_α and Π_fuzzy, the retained mass and the number of
communities. The earlier version evaluated the shortcut at the single
Algorithm-1 resolution of Table 5, which on the 2009 item-level graph
at α = 0.6 produced a loss of 0.93; refitting γ reduces that figure to
0.35, so about two thirds of the earlier number was a resolution
artefact, as the referee anticipated, and the remaining third is the
loss proper.

**Table 8.** The binarise-then-partition shortcut on the six real
graphs, resolution refitted per level set. "Fuzzy" gives Q_F, the
refitted γ and the number of communities of Fuzzy Leiden. Each α cell
gives loss(α) / NMI(Π_α, Π_fuzzy) / share of fuzzy mass retained /
communities. A dash means no edge survives the cut.

  ----------------------------------------------------------------------------------------------------------------------
  Graph              Fuzzy (Q_F; γ; k)   α = 0.2                 α = 0.4                 α = 0.6                 α = 0.8
  ------------------ ------------------- ----------------------- ----------------------- ----------------------- -----------------------
  MOOCCubeX-CS       0.524; 1; 7        0.002 / 0.81 / 1.00 / 6 0.003 / 0.85 / 1.00 / 7 0.003 / 0.88 / 0.99 / 7 0.016 / 0.81 / 0.97 / 10

  2009, KC level     0.102; 1; 5        -0.000 / 0.73 / 1.00 / 40.003 / 0.65 / 0.79 / 5 0.067 / 0.30 / 0.25 / 15---

  2009, item level   0.873; 0.25; 28       0.000 / 1.00 / 1.00 / 280.001 / 0.99 / 0.84 / 300.351 / 0.80 / 0.10 / 169---

  2015, KC level     0.059; 1; 3        0.002 / 0.53 / 1.00 / 3 0.002 / 0.62 / 0.73 / 3 0.054 / 0.07 / 0.13 / 6 ---

  2017, KC level     0.025; 1; 3        0.004 / 0.49 / 1.00 / 3 0.003 / 0.42 / 0.90 / 2 0.027 / 0.16 / 0.03 / 17---

  2017, item level   0.564; 1; 19       0.000 / 0.98 / 1.00 / 19-0.000 / 0.99 / 0.85 / 200.079 / 0.74 / 0.20 / 44---
  ----------------------------------------------------------------------------------------------------------------------

The real graphs tell the same story in retained mass. At α = 0.2 the
level set is the whole support graph, since the construction prunes at
τ_E = 0.2, and at α = 0.4 the cut still keeps 73--90% of the mass: the
loss is at most 0.004 on every graph and the item-level partitions
agree with the fuzzy ones at NMI 0.98--1.00. At α = 0.6 the cut keeps
3--25% of the mass and the loss is 0.35 on the 2009 item-level graph,
0.08 on the 2017 item-level graph and 0.03--0.07 on the KC-level
graphs, with NMI against the fuzzy partition falling to 0.07--0.30 on
the KC-level graphs, where the fuzzy partition is itself not
structure (Table 7). On MOOCCubeX-CS every cut up to α = 0.8 keeps at
least 97% of the mass and the loss never exceeds 0.016, while on the
ASSISTments graphs no edge survives α = 0.8; these are statements about
the mean membership, which decides how much mass an absolute level
keeps, not about the spread. The Corollary 1 identity was evaluated on every graph at the
same resolution, the gap equalling γ Σ_κ Cov(f_κ, g_κ) to four
decimals and ranging from 0.004 (MOOCCubeX-CS) to 0.339 (2017 item
level); the Cauchy--Schwarz bound lies within 0.001 of it on all six
graphs because the graphs are near-reciprocal (up to 0.998), so f_κ ≈
g_κ and the bound is an identity up to the asymmetry. The gap of 0.042
on the 2009 item-level graph next to a shortcut loss of 0.93 is the
real-data instance of Remark 3.

### 9.2 Scalability and stability

The real graphs above establish that Fuzzy Leiden runs correctly; they
do not establish how it scales. Synthetic fuzzy prerequisite graphs were
generated to match the MOOCCubeX-CS structural statistics: density fixed
at \|E\|/\|C\| ≈ 1.54, 40 topics per 1000 nodes, edges placed within a
topic with probability p_within and across topics otherwise, directions
constrained to respect a randomly drawn curriculum order so the graph is
acyclic, and memberships drawn from Beta(2.0, 1.2) clipped to \[0.2, 1\]
so that the τ_E pruning is respected by construction. Two regimes,
p_within = 0.85 (strongly modular) and 0.55 (weakly modular), and five
seeds per size.

**Table 9.** Fuzzy Leiden on synthetic fuzzy graphs, γ = 0.5, mean of 5
seeds. "Conn." is the Lemma 1(i) check on every returned community;
"Mono." is the Lemma 1(iv) check over 12 successive optimisation
iterations.

  -------------------------------------------------------------------------------------
  p_within   \|C\|    \|E\|    Time (s) Q_F     sd Q_F  Comms   Max.    Conn.   Mono.
                                                                share           
  ---------- -------- -------- -------- ------- ------- ------- ------- ------- -------
  0.85       59       91       0.004    0.712   0.029   6.6     33.6%   pass    pass

  0.85       250      385      0.019    0.834   0.012   21.0    15.8%   pass    pass

  0.85       1,000    1,540    0.104    0.876   0.007   64.0    8.5%    pass    pass

  0.85       5,000    7,700    0.911    0.897   0.003   258.6   4.5%    pass    pass

  0.85       10,000   15,400   4.214    0.901   0.001   499.8   3.2%    pass    pass

  0.85       20,000   30,800   12.702   0.905   0.001   943.2   2.4%    pass    pass

  0.55       59       91       0.004    0.662   0.006   8.0     34.2%   pass    pass

  0.55       250      385      0.028    0.733   0.006   20.4    19.3%   pass    pass

  0.55       1,000    1,540    0.178    0.760   0.004   61.4    11.4%   pass    pass

  0.55       5,000    7,700    2.222    0.773   0.002   270.6   6.0%    pass    pass

  0.55       10,000   15,400   6.945    0.777   0.002   515.2   5.0%    pass    pass

  0.55       20,000   30,800   17.047   0.779   0.001   973.4   3.9%    pass    pass
  -------------------------------------------------------------------------------------

A log-log fit of runtime against edge count from 250 nodes upward gives
empirical exponents of 1.48 and 1.49 in the two regimes, above the
O(\|E\| log \|C\|) per-iteration bound. The number of communities grows
roughly linearly in \|C\| on these graphs (Table 9), and we attribute
the excess to the number of aggregation levels traversed; the released
script records that number per run so the attribution can be checked.
The density of these graphs, about 1.5 edges per node, is that of the
curated graph and not of the KC-level graphs (about 50), so Table 9
says nothing about scaling in the dense regime. A 20000-node graph is
partitioned in 12.7 s in the strongly modular regime and 17.0 s in the
weakly modular one.

Seed-to-seed dispersion of Q_F *falls* with graph size, from 0.029 at 59
nodes to 0.001 at 20000 in the strongly modular regime and from 0.006 to
0.001 in the weakly modular one. Small graphs are the unstable regime
for modularity optimisation, which is a caution about the 59-node result
of Table 2 as much as a reassurance about scale.

Lemma 1(i) was checked on the seed-0 partition of every size and regime,
3523 communities in total across the twelve configurations, and was not
violated; the released script records the check per configuration. For Lemma 1(iv), Q_F was recorded at each of 12
successive optimisation iterations for every size, regime and seed,
giving 720 successive differences, none of which was negative (minimum
difference greater than −10\^{−9}).

What Table 9 supports is a claim about the optimiser, not about real
graphs of that size: a generator with planted topics produces structure
that is easier to recover than a real curriculum's.

### 9.3 Alternative partitioners and composition operators

**Partitioners.** Table 10 compares Fuzzy Leiden against two baselines on
the identical MOOCCubeX-CS graph, all evaluated under the same objective
Q_F (γ = 0.5) for comparability: spectral clustering on the
fuzzy-weighted adjacency matrix requesting the same number of clusters
Fuzzy Leiden found, and fuzzy c-means applied to the leading
eigenvectors of the normalised graph Laplacian, a standard
spectral-embedding-plus-fuzzy-clustering construction, no independent
node features existing beyond graph structure for this graph.

**Table 10.** Partitioners on the real MOOCCubeX-CS graph, all evaluated
under Q_F (γ = 0.5), best of 10 seeds each.

  -----------------------------------------------------------------------
  Method            Q_F               Cluster sizes     Note
  ----------------- ----------------- ----------------- -----------------
  Fuzzy Leiden      0.672             {21,20,16,2}      Best objective

  Spectral          0.664             {30,21,6,2}       Close, one
  clustering (k =                                       dominant cluster
  4)                                                    

  Fuzzy c-means on  0.639             {21,21,9,8}       Balanced, lowest
  spectral                                              objective
  embedding (c = 4)                                     
  -----------------------------------------------------------------------

All three methods land within 0.04 of each other on this graph of 59
nodes and 91 edges, which Section 9.2 identifies as the unstable
regime, so the comparison is indicative only; Fuzzy Leiden is ahead; the spectral baseline concentrates half the
graph into one cluster to get there, while the hardened fuzzy c-means
partition is the most balanced and the furthest from the optimum. The
raw sizes are reported rather than only the objective so that the
trade-off is visible: closeness in Q_F does not imply interchangeability
for a downstream use that summarises communities individually.
Soft-modularity optimisation \[10\] and fuzzy modularity maximisation
\[12,13\] return graded memberships rather than a partition; hardening
their output changes what they optimise, so that comparison is left to
the released code as a documented option.

**Composition operators.** Table 11 sweeps both parametric families on
the real ASSISTments 2009 item-level fuzzy graph of Table 3 (371 nodes,
4950 directed fuzzy edges, mean membership 0.421) over 1529 ordered node
pairs reachable within four hops from 60 sampled sources, taking for
each pair the exact best path under each operator. The best paths were
computed by bounded enumeration in the earlier version and are exact
within the four-hop limit; the released script now computes them by
the generator-based shortest path of Section 7.3, which the enumeration
agrees with to 10\^{−15} on every family and every target checked, and
without a hop limit, so the four-hop truncation can be dropped. Two derived quantities
accompany the raw strengths. *Erosion* is E = mean\[(S_min −
S_T)/S_min\], zero for the minimum by construction and positive for
every other t-norm by Fact 1. *Rank agreement* ρ is Spearman's
correlation between the operator's ordering of pairs and the minimum's,
which is the quantity that matters downstream because the path score of
Equation (16) consumes an ordering rather than absolute strengths.

**Table 11.** Parametric t-norm families and compensatory alternatives
on the real ASSISTments 2009 item-level fuzzy graph, 1529 reachable
pairs within four hops. The pooled Spearman ρ of the earlier version is
kept for continuity; Table 11b adds the per-source Kendall τ-b with
95% bootstrap intervals over sources, computed with the exact
shortest-path solutions of Section 7.3 by the repository pipeline (see
Data availability).

  -----------------------------------------------------------------------------------
  Operator           Class          Mean S  2-hop   3-hop   4-hop   Erosion E ρ vs
                                                                              min
  ------------------ -------------- ------- ------- ------- ------- --------- -------
  min (default)      t-norm         0.450   0.445   0.411   0.395   0.000     1.000

  product            t-norm         0.275   0.201   0.094   0.047   0.407     0.591

  Łukasiewicz        t-norm         0.202   0.012   0.000   0.000   0.574     0.529

  Yager p = 1.5      t-norm         0.234   0.125   0.003   0.000   0.504     0.560

  Yager p = 2        t-norm         0.272   0.216   0.064   0.003   0.420     0.621

  Yager p = 3        t-norm         0.333   0.304   0.210   0.142   0.274     0.707

  Yager p = 5        t-norm         0.385   0.371   0.316   0.278   0.149     0.837

  Yager p = 10       t-norm         0.423   0.417   0.380   0.357   0.059     0.952

  Schweizer--Sklar p t-norm         0.423   0.415   0.382   0.364   0.058     0.946
  = −8                                                                        

  Schweizer--Sklar p t-norm         0.394   0.379   0.340   0.317   0.124     0.839
  = −4                                                                        

  Schweizer--Sklar p t-norm         0.360   0.331   0.277   0.247   0.204     0.719
  = −2                                                                        

  Schweizer--Sklar p t-norm         0.329   0.286   0.213   0.175   0.276     0.649
  = −1                                                                        

  Schweizer--Sklar p t-norm         0.275   0.201   0.094   0.047   0.407     0.591
  → 0 (= product)                                                             

  Schweizer--Sklar p t-norm         0.233   0.118   0.006   0.000   0.506     0.574
  = 0.5                                                                       

  Schweizer--Sklar p t-norm         0.202   0.012   0.000   0.000   0.574     0.529
  = 1                                                                         

  geometric mean     compensatory   0.497   0.498   0.480   0.462   −0.117    0.879

  arithmetic mean    compensatory   0.500   0.502   0.485   0.467   −0.126    0.856

  path-level OWA     compensatory   0.477   0.475   0.453   0.437   −0.068    0.946
  (pessimistic)                                                               
  -----------------------------------------------------------------------------------

**Table 11b.** Per-source rank agreement with max-min on the same
graph: Kendall τ-b between the ordering of reachable targets by exact
best-path strength under each operator and under the minimum, mean
over 51 sources with at least three reachable targets (of 60
sampled), with 95% bootstrap intervals over sources; with the four-hop
limit of Table 11 and without it (1507 source--target pairs pooled).

  ------------------------------------------------------------------------------------------------
  Operator                   τ-b, 4 hops   95% CI            τ-b, no hop limit  95% CI            pooled ρ, no limit
  -------------------------- ------------- ----------------- ------------------ ----------------- ------------------
  product                    0.306         [0.253, 0.365]    0.303              [0.248, 0.362]    0.514

  Łukasiewicz                0.278         [0.230, 0.330]    0.277              [0.229, 0.329]    0.465

  Yager p = 1.5              0.309         [0.254, 0.365]    0.307              [0.252, 0.363]    0.521

  Yager p = 2                0.332         [0.276, 0.389]    0.329              [0.274, 0.385]    0.587

  Yager p = 3                0.395         [0.341, 0.453]    0.395              [0.340, 0.454]    0.700

  Yager p = 5                0.504         [0.449, 0.561]    0.512              [0.457, 0.569]    0.827

  Yager p = 10               0.615         [0.563, 0.668]    0.640              [0.587, 0.690]    0.929

  Schweizer--Sklar p = −8    0.543         [0.489, 0.596]    0.556              [0.500, 0.610]    0.861

  Schweizer--Sklar p = −4    0.443         [0.387, 0.501]    0.449              [0.392, 0.507]    0.746

  Schweizer--Sklar p = −2    0.366         [0.316, 0.423]    0.366              [0.315, 0.422]    0.641

  Schweizer--Sklar p = −1    0.333         [0.279, 0.391]    0.330              [0.276, 0.390]    0.577

  Schweizer--Sklar p = 0.5   0.295         [0.244, 0.352]    0.292              [0.241, 0.350]    0.490
  ------------------------------------------------------------------------------------------------


The bound of Fact 1 was checked directly rather than assumed: over every
enumerated path and all fifteen t-norm rows of the table (two of which,
Yager p = 1 and Schweizer--Sklar p = 1, coincide with Łukasiewicz, and
the p → 0 row of the Schweizer--Sklar family with the product), no
t-norm aggregate exceeded the weakest link, with zero violations. The
three compensatory operators are not t-norms, and their negative erosion
is the point --- they sit above the weakest link, which no t-norm can.

Three consequences follow, in descending order of importance.

First, the operator choice is more consequential than the pooled
statistic suggested. Pooled rank agreement ranges from 0.529
(Łukasiewicz, equivalently Yager p = 1 and Schweizer--Sklar p = 1) to
0.952 (Yager p = 10), but that statistic pools pairs across sources
and hop lengths and is partly a hop-length and tie effect, since
Łukasiewicz returns 0 for most 3- and 4-hop pairs, whereas the
ordering Equation (16) consumes is within a source. The per-source
Kendall τ-b of Table 11b is the relevant quantity, and it is far
lower: 0.28 for Łukasiewicz,
0.30 for the product, and at most
0.64 (Yager p = 10) at the near-min end, with bootstrap intervals
of width about 0.1. Within a source, every Archimedean operator
reorders the candidates substantially on this graph; the hop limit
changes the figures by at most 0.03, so the earlier four-hop
truncation did not bias them. And the comparison with MOOCCubeX-CS, whose
mean membership is 0.934 and where agreement never falls below 0.931
over 316 reachable pairs, does not identify the mean as the cause,
since the two graphs differ in size, density, acyclicity and hop-length
distribution as well; a mean near 1 compresses every Archimedean
operator toward the minimum regardless of spread, which is an algebraic
fact rather than a measured one, and the earlier version's phrase
"depends almost entirely on the spread" conflated mean with spread. The
controlled statement is Proposition 3 and its verification below: the
reordering is a consequence of the operators' sensitivity to the
membership scale, and it is absent for max-min by construction.

Second, agreement is high in the near-min region of both families and
falls away from it. For Yager p = 10 and Schweizer--Sklar p = −8 the
ordering agrees with the minimum at ρ ≥ 0.94, and it decays smoothly
toward the Łukasiewicz end, so a deployment wanting a mild hop-length
penalty can move a little way into either family without much
reordering. Erosion grows monotonically as p moves away from the
min-limit and concentrates on long paths, so within the class p is a
dial on hop-length penalty, never on compensation, exactly as Fact 1
requires.

Third, the cost of the alternatives is that of a shortest-path sweep,
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
316 pairs reachable within five hops are used.

**Table 12.** Mean fuzzy path strength by hop count, real MOOCCubeX-CS
graph, 316 reachable pairs.

  --------------------------------------------------------------------------
  Hops           n pairs        min            product        Łukasiewicz
  -------------- -------------- -------------- -------------- --------------
  1              91             0.934          0.934          0.934

  2              106            0.878          0.860          0.857

  3              68             0.880          0.837          0.830

  4              35             0.863          0.808          0.797

  5              16             0.854          0.780          0.763
  --------------------------------------------------------------------------

Max-min strength is roughly flat beyond one hop and non-monotonic,
driven by which single edge is weakest on a given path, while the
product and Łukasiewicz decrease consistently as hop count grows,
accumulating the effect of every edge. Neither behaviour is
unconditionally correct: the minimum answers how strong the weakest
prerequisite on a path is, which is the question Equation (16) is built
around and which is well posed on an ordinal scale, while the
Archimedean t-norms answer how much cumulative erosion a long chain
suffers, which presupposes a cardinal one.

### 9.4 Verification of the fuzzification and sensitivity of the construction parameters

*Orness of the edge aggregation.* The OWA weights w = (0.5, 0.3, 0.2) of
Equation (15) have orness 0.65. Sweeping weight vectors with orness from
0.50 to 0.80 and rebuilding three graphs leaves the retained edge count
nearly flat (2009 KC level: 4827 to 5074 edges, within ±2.4% of the
default 5059; 2009 item level: 4885 to 4958; 2017 item level: 32614 to
33674) while the mean membership moves linearly (0.376 to 0.569, 0.339
to 0.508 and 0.364 to 0.549 respectively). The pruning threshold τ_E =
0.2 is therefore insensitive to the orness; what the orness sets is the
operating point of the membership scale, which is exactly where Section
8.3 showed the composition-operator choice acquires or loses its
consequence. The two parameters should be tuned together.
*Fuzzification.* Equations (11)--(12) were checked over a grid of 505
combinations --- 101 point estimates spanning \[0,1\] in steps of 0.01,
crossed with five uncertainty widths ησ ∈ {0, 0.05, 0.10, 0.20, 0.30}
--- against the three properties claimed for them. Zero violations were
found for sum-to-one (10\^{−5} floating-point tolerance), for membership
range in \[0,1\] and for monotonicity of μ_mas (non-decreasing) and
μ_unk (non-increasing) in the point estimate. Three limiting cases
behave as designed: p̄\^\* = 0.95 with zero uncertainty yields (0, 0, 1),
p̄\^\* = 0.05 yields (1, 0, 0) and p̄\^\* = 0.60 = α_par yields (0, 1, 0).

Each cut-point was then perturbed over {−0.10, −0.05, 0, +0.05, +0.10},
giving 125 settings, all of which remain well-ordered and were therefore
all evaluated on the same 505-point grid, for 63125 fuzzifications. All
three properties hold under every setting, with zero violations
throughout: the correctness of the construction is not contingent on the
particular cut-points. Table 14 reports the L¹ distance between the
perturbed and baseline triples and the rate at which the *defuzzified*
label --- the argmax of the triple, which is what a downstream gate acts
on --- flips.

**Table 14.** Cut-point sensitivity of the fuzzification, 505-point grid
per setting.

  -----------------------------------------------------------------------
  Perturbation      Mean L¹           Max L¹            Band flips
  ----------------- ----------------- ----------------- -----------------
  α_unk − 0.10      0.075             0.571             4.8%

  α_unk − 0.05      0.037             0.333             2.4%

  α_unk + 0.05      0.037             0.400             3.0%

  α_unk + 0.10      0.068             0.800             6.3%

  α_par − 0.10      0.137             0.800             4.6%

  α_par − 0.05      0.070             0.400             2.4%

  α_par + 0.05      0.070             0.500             3.0%

  α_par + 0.10      0.139             1.000             5.5%

  α_mas − 0.10      0.065             1.000             3.2%

  α_mas − 0.05      0.033             0.500             1.8%

  α_mas + 0.05      0.037             0.400             2.2%

  α_mas + 0.10      0.075             0.667             3.4%

  All three         0.132             ---               6.0%
  jointly, mean of                                      
  125 settings                                          

  All three         0.262             ---               13.9%
  jointly, worst                                        
  setting                                               
  -----------------------------------------------------------------------

Three points follow. The middle cut-point α_par is the sensitive
parameter and α_mas the insensitive one: moving α_par by ±0.10 shifts
the triple roughly twice as far in L¹ as moving either outer cut-point
by the same amount (0.137--0.139 against 0.065--0.075), which is
structural, since α_par appears in the denominator of both μ_unk and
μ_mas in (11) and therefore rescales both transition bands at once. A
practitioner calibrating for a new domain should spend the
expert-labelling budget on the competence threshold.

The defuzzified band is more robust than the membership: the worst
single shift of ±0.10 changes it at 6.3% of grid points and the worst
joint shift at 13.9%. The gate of Section 8.3 tests μ_par against
τ_ZPD rather than taking the argmax, so band flips are not gate flips;
the released script reports both. The graded triple moves more
than the band does because most of the L¹ movement occurs strictly
inside a band rather than across its boundary, which is the intended
behaviour of a graded representation and an argument for the fuzzy
formulation: a crisp three-band classifier with the same cut-points
would expose the full movement as a label change.

Robustness of the representation is not robustness of the decision that
consumes it. The band flips concentrate near the boundaries, which is
exactly where the gate of Equation (16) is most likely to be decisive.
What Table 14 bounds is the input-side perturbation, not its downstream
propagation.

### 9.5 Gates on the uncertain state

The evaluation grid is 201 point estimates spanning \[0,1\] crossed
with seven half-widths ησ ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30},
giving 1407 states; the grid is the author's choice, and the
percentages below are properties of it, not of any population of
learners. The footprint of Equation (13) is well formed on every
state (no μ̲ \> μ̄, no membership outside \[0,1\], a point at ησ =
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

The cautious collapse admits nothing from ησ = 0.10, as h\^\* = 0.075
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

### 9.6 Component latency

Component latency on a single core, no GPU: Fuzzy Leiden on the 59-node
MOOCCubeX-CS graph 3.4 ms per run (offline, once per corpus, mean of 5
runs); an 8-pass Monte-Carlo-dropout forward pass of the encoder architecture
of Section 8.2 over a 20-step history 79.7 ms (online, mean of 20 runs;
the timing was taken on an initialised network of the stated
architecture, since the forward cost does not depend on the weights,
and no trained encoder is evaluated in this paper); footprint
construction for 1407 states 0.09 ms (Section 9.5). The optimiser and
the fuzzification are therefore not where a deployed pipeline spends
its time; the encoder is.

## 10. Discussion

### 10.1 What the three costs turn out to be

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
collapse rejects every candidate once the half-width passes h\^\* =
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

### 10.2 What is out of scope

This paper prices three shortcuts; it does not evaluate a recommender,
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
measurements of what was measured.

### 10.3 Limitations

*Fuzzy edges, crisp partitions.* Equation (7) fuzzifies one argument of
the modularity functional. The functional fuzzifying both is written
down in Appendix A: its aggregation step is exact only along
constant-membership blocks (Proposition A.1), so the obstruction to a
fast multi-level optimiser is located, and what remains open is an
optimiser for Q\_{FF} that is either exact on a useful stratum or
approximate with a bounded loss.

*Proposition 2 is a normalisation statement.* It bounds the α-average
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
Stability across γ, resolution profiles \[46\], consensus clustering
\[48\] and resolution-limit-free objectives \[47\] address the same
problem; the null-model step is the only external check reported, and
a comparison on planted benchmarks with varying density is left open.

*Scale is established synthetically.* Table 9 establishes how the
optimiser scales, not how a real graph of that size partitions. Real
educational knowledge graphs at 20000 nodes have hub concepts,
editorially introduced cycles and unequal topic sizes that the generator
does not produce, and the modularity level on them is not predicted by
the synthetic figures.

*The cut-point analysis bounds the input, not the outcome.* Section 9.4
shows that a ±0.10 mis-specification leaves the defuzzified band
unchanged on 86% of the state space in the worst joint setting and on
94--98% under single shifts; it does not show what the remaining flips
do downstream, because they concentrate near the decision boundary.

*The gates on the uncertain state are compared on a grid, not on
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
memberships is the natural extension and is not reported.

## 11. Conclusion

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

## CRediT authorship contribution statement

**Hai Bang Truong:** Conceptualization, Methodology, Formal analysis,
Software, Validation, Investigation, Data curation, Writing -- original
draft, Writing -- review & editing, Visualization.

## Declaration of competing interest

The author declares that he has no known competing financial interests
or personal relationships that could have appeared to influence the work
reported in this paper.

## Funding

This work was carried out as part of the author's academic duties at
Saigon International University. No external funding was received.

## Data availability

All datasets used in this paper are public. The ASSISTments 2009
skill-builder release is available via the ASSISTments Data Portal and
mirrored on figshare (article 25309000); the 2015 and 2017 releases are
available via the ASSISTments Data Portal and the EduData mirror at
base.ustc.edu.cn/data/ASSISTment/. The MOOCCubeX corpus \[3\] is
released by THU-KEG at https://github.com/THU-KEG/MOOCCubeX, with the
concept prerequisite graph (prerequisites/cs.json) and the
concept--problem mapping (relations/concept-problem.txt) served from the
MOOCCubeX distribution. A single self-contained script that downloads
nothing but these public files, rebuilds every fuzzy graph of Tables
2--3 with the stated preprocessing, reproduces every table, runs the planted benchmark of Table 6, the
null-model test of Table 7, the generator-based best paths and the
ordinal-invariance check of Section 9.3 and the gate comparison of Table
13, and compares the printed values listed in the supplement against
its own output, is maintained at
https://github.com/Bangth2026/fuzzy-kg-modularity. \[CẦN ĐỐI CHIẾU
TRƯỚC KHI NỘP: pipeline trong repo dựng lại sáu đồ thị với cùng số
nút và số cạnh sai khác dưới 0.3% so với Bảng 3 (MOOCCubeX-CS trùng
hoàn toàn), nhưng trung bình μ_E cao hơn Bảng 3 từ 0.01 đến 0.05 (ví
dụ 2009 item: 0.470 so với 0.421) và bước spectral bisection cho c/W
khác Bảng 5; nghĩa là `16_run_experiments.py` dùng cho Bảng 2--5, 11
và mã trong repo chưa đồng nhất về định nghĩa mastery gain/OWA và về
bisection. Bảng 7 (sáu dòng thật), Bảng 8 và Bảng 11b trong bản này
tính bằng pipeline repo. Cần chọn một pipeline làm chuẩn và tính lại
các bảng còn lại bằng nó.\]

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author used a generative AI
assistant to assist with language editing and with implementing and
executing the data-processing and experiment code whose outputs are
reported in Section 9. Every reported empirical value is produced by the
released pipeline from the cited public datasets, with no value entered
by hand; the pipeline is provided as supplementary material so that
every table can be reproduced independently. The author reviewed the
code and the results and takes full responsibility for the content of
the published article.

## Acknowledgements

The author thanks the ASSISTments team at Worcester Polytechnic
Institute for maintaining and releasing the datasets, and the pyKT
community for reference implementations.

## Appendix A. A functional for fuzzy partitions of fuzzy graphs

This appendix records the simultaneous generalisation requested by the
fuzzy-partition line and analyses the property that decides whether a
Leiden-type optimiser can serve it. Let a *fuzzy partition* of C into M
communities be a matrix Ũ = (u\_{km}) with u\_{km} ≥ 0 and Σ_m u\_{km} =
1 for every k ∈ C, and write ⟨u_u, u_v⟩ = Σ\_{m=1}\^{M} u\_{um} u\_{vm}
for the coupling of two membership rows. Define, with the conventions of
Section 6,

$$Q_{FF}\left( \widetilde{U} \right)\mspace{6mu} = \mspace{6mu}\frac{1}{W}\sum_{u,\, v\, \in \, C}^{}\left\lbrack \mu_{E}(u,v)\mspace{6mu} - \mspace{6mu}\gamma\,\frac{s^{out}(u)\, s^{in}(v)}{W} \right\rbrack\langle u_{u},u_{v}\rangle.\quad\quad\text{(A.1)}$$

Equation (A.1) replaces the Kronecker delta of (7) by the bilinear
coupling of \[9,10\], so both fuzzifications act on their own argument
of the same functional.

**Proposition A.1.** *Let G_F be a directed fuzzy graph with W \> 0 and
γ \> 0. Then:*

*(i) Reductions. If Ũ is crisp (each row a vertex of the simplex),
Q\_{FF} = Q_F of Equation (7). If μ_E is crisp, Q\_{FF} is the directed
form of the soft modularity of \[9,10\]. If both are crisp and γ = 1,
Q\_{FF} is the directed modularity of \[35\].*

*(ii) Boundedness. Q\_{FF}(Ũ) ∈ \[−γ, 1\] for every fuzzy partition Ũ.*

*(iii) Aggregation. Let B ⊆ C and let G_F/B contract B to one node b
carrying self-loop mass W_B, out-strength S_B\^{out}, in-strength
S_B\^{in} and membership row u_b. If all rows in B are equal, u_k = u_B
for k ∈ B, then setting u_b = u_B makes Q\_{FF} invariant under the
contraction. Conversely, if invariance is required for every graph
extending G_F on the same vertex set, the rows in B must be equal.*

*Proof.* (i) For crisp rows ⟨u_u, u_v⟩ = δ(κ(u), κ(v)); the remaining
reductions follow from Lemma 1(vi). (ii) Each row lies on the
probability simplex, so ‖u_k‖₂ ≤ 1 and 0 ≤ ⟨u_u, u_v⟩ ≤ 1 by
Cauchy--Schwarz. The membership term of (A.1) then lies in \[0, 1\] and
the null term in \[0, γ\], since Σ\_{u,v} s\^{out}(u) s\^{in}(v) = W².
(iii) If the rows in B are equal, every coupling involving a node of B
factors through u_B: for u ∈ B and v ∉ B, ⟨u_u, u_v⟩ = ⟨u_B, u_v⟩, and
for u, v ∈ B, ⟨u_u, u_v⟩ = ⟨u_B, u_B⟩, so the internal terms sum to
\[W_B − γ S_B\^{out} S_B\^{in}/W\] ⟨u_B, u_B⟩/W and the cross terms to
the contracted expression, term by term. For the converse, take u₁, u₂ ∈
B with u\_{u₁} ≠ u\_{u₂} and extend the graph by an external node w
carrying an edge to u₁ only; invariance for all extensions forces ⟨u_w,
u\_{u₁}⟩ = ⟨u_w, u_b⟩ for every row u_w on the simplex, hence u_b =
u\_{u₁}, and the symmetric extension with an edge to u₂ forces u_b =
u\_{u₂}, a contradiction. ∎\*

The consequence for algorithms is the point of the exercise. The
multi-level scheme that makes Leiden fast is exact for Q_F because of
Lemma 1(iii); for Q\_{FF} it is exact only when contraction is
restricted to constant-membership blocks, so a Leiden-style optimiser
for Q\_{FF} is exact only if its refinement phase returns crisp cores,
and is otherwise an approximation. This explains, from the present
standpoint, why the fuzzy-partition literature optimises membership
matrices by local iterative updates rather than by contraction
hierarchies \[10,11,13\], and it makes precise what Section 10.3 states
informally: aggregation-invariance is the property to establish, and for
the bilinear coupling it holds exactly on, and only on, the crisp-core
stratum.

## References

\[1\] A. Rosenfeld, Fuzzy graphs, in: L.A. Zadeh, K.S. Fu, K. Tanaka, M.
Shimura (Eds.), Fuzzy Sets and Their Applications to Cognitive and
Decision Processes, Academic Press, New York, 1975, pp. 77--95.

\[2\] X. Cheng, Z. Zhang, J. Wang, L. Fang, C. He, Q. Guan, S. Pan, W.
Luo, GraphRAG-induced dual knowledge structure graphs for personalized
learning path recommendation, Proc. AAAI Conf. Artif. Intell. 40 (2026)
14610--14620. https://doi.org/10.1609/aaai.v40i17.38479

\[3\] J. Yu, Y. Wang, Q. Zhong, G. Luo, Y. Mao, K. Sun, W. Feng, W. Xu,
S. Cao, K. Zeng, Z. Yao, L. Hou, Y. Lin, P. Li, J. Zhou, B. Xu, J. Li,
J. Tang, M. Sun, MOOCCubeX: a large knowledge-centered repository for
adaptive learning in MOOCs, in: Proc. 30th ACM Int. Conf. Information
and Knowledge Management, 2021, pp. 4643--4652.

\[4\] K.R. Bhutani, A. Rosenfeld, Strong arcs in fuzzy graphs, Inf. Sci.
152 (2003) 319--322.

\[5\] M.E.J. Newman, Modularity and community structure in networks,
Proc. Natl. Acad. Sci. USA 103 (2006) 8577--8582.

\[6\] V.D. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre, Fast
unfolding of communities in large networks, J. Stat. Mech. 2008 (2008)
P10008.

\[7\] V.A. Traag, L. Waltman, N.J. van Eck, From Louvain to Leiden:
guaranteeing well-connected communities, Sci. Rep. 9 (2019) 5233.
https://doi.org/10.1038/s41598-019-41695-z

\[8\] E.P. Klement, R. Mesiar, E. Pap, Triangular Norms, Kluwer Academic
Publishers, Dordrecht, 2000.

\[9\] T. Nepusz, A. Petróczi, L. Négyessy, F. Bazsó, Fuzzy communities
and the concept of bridgeness in complex networks, Phys. Rev. E 77
(2008) 016107. https://doi.org/10.1103/PhysRevE.77.016107

\[10\] T.C. Havens, J.C. Bezdek, C. Leckie, K. Ramamohanarao, M.
Palaniswami, A soft modularity function for detecting fuzzy communities
in social networks, IEEE Trans. Fuzzy Syst. 21 (2013) 1170--1175.

\[11\] H.-W. Su, T.C. Havens, Quadratic program-based modularity
maximization for fuzzy community detection in social networks, IEEE
Trans. Fuzzy Syst. 23 (2015) 1356--1371.

\[12\] S. Gregory, Fuzzy overlapping communities in networks, J. Stat.
Mech. 2011 (2011) P02017.

\[13\] S. Yazdanparast, T.C. Havens, M. Jamalabdollahi, Soft overlapping
community detection in large-scale networks via fast fuzzy modularity
maximization, IEEE Trans. Fuzzy Syst. 29 (2021) 1533--1543.

\[14\] J. Xiao, Y. Wang, X. Xu, Fuzzy community detection based on elite
symbiotic organisms search and node neighborhood information, IEEE
Trans. Fuzzy Syst. 30 (2022) 2500--2514.

\[15\] K. Zhou, A. Martin, Q. Pan, Z.-G. Liu, Median evidential c-means
algorithm and its application to community detection, Knowl.-Based Syst.
74 (2015) 69--88.

\[16\] J. Xiao, X.-K. Xu, Community detection from fuzzy and
higher-order perspectives, EPL 144 (2023) 11003.
https://doi.org/10.1209/0295-5075/acfdc9

\[17\] M. Barroso, D. Gómez González, I. Gutiérrez García-Pardo, A
supervised approach to community detection problem: how to improve
Louvain algorithm by considering fuzzy measures, in: Intelligent and
Fuzzy Systems (INFUS 2022), Lecture Notes in Networks and Systems 504,
Springer, Cham, 2022, pp. 219--227.
https://doi.org/10.1007/978-3-031-09173-5_28

\[18\] I. Gutiérrez García-Pardo, M. Barroso Pérez, D. Gómez González,
J. Castro Cantalejo, Improving community detection algorithms in
directed graphs with fuzzy measures. An application to mobility
networks, Expert Syst. Appl. 269 (2025) 126305.
https://doi.org/10.1016/j.eswa.2024.126305

\[19\] S. Fortunato, M. Barthélemy, Resolution limit in community
detection, Proc. Natl. Acad. Sci. USA 104 (2007) 36--41.
https://doi.org/10.1073/pnas.0605965104

\[20\] S. Fortunato, M.E.J. Newman, 20 years of network community
detection, Nat. Phys. 18 (2022) 848--850.
https://doi.org/10.1038/s41567-022-01716-7

\[21\] J.M. Mendel, R.I.B. John, Type-2 fuzzy sets made simple, IEEE
Trans. Fuzzy Syst. 10 (2002) 117--127.

\[22\] N.N. Karnik, J.M. Mendel, Centroid of a type-2 fuzzy set, Inf.
Sci. 132 (2001) 195--220.

\[23\] M. Nie, W.W. Tan, Towards an efficient type-reduction method for
interval type-2 fuzzy logic systems, in: Proc. IEEE Int. Conf. Fuzzy
Systems, 2008, pp. 1425--1432.

\[24\] D. Wu, J.M. Mendel, Enhanced Karnik--Mendel algorithms, IEEE
Trans. Fuzzy Syst. 17 (2009) 923--934.

\[25\] K. Almohammadi, H. Hagras, D. Alghazzawi, G. Aldabbagh, A type-2
fuzzy logic recommendation system for adaptive teaching, Soft Comput. 21
(2017) 965--979.

\[26\] Y. Gal, Z. Ghahramani, Dropout as a Bayesian approximation:
representing model uncertainty in deep learning, in: Proc. 33rd Int.
Conf. Machine Learning, 2016, pp. 1050--1059.

\[27\] D. Dubois, L. Foulloy, G. Mauris, H. Prade,
Probability--possibility transformations, triangular fuzzy sets, and
probabilistic inequalities, Reliab. Comput. 10 (2004) 273--297.

\[28\] S. Mathew, M.S. Sunitha, Types of arcs in a fuzzy graph, Inf.
Sci. 179 (2009) 1760--1768.

\[29\] R.R. Yager, On a general class of fuzzy connectives, Fuzzy Sets
Syst. 4 (1980) 235--242.

\[30\] B. Schweizer, A. Sklar, Probabilistic Metric Spaces,
North-Holland, New York, 1983.

\[31\] K. Chrysafiadi, M. Virvou, Student modeling approaches: a
literature review for the last decade, Expert Syst. Appl. 40 (2013)
4715--4729.

\[32\] L.A. Zadeh, Fuzzy sets, Inf. Control 8 (1965) 338--353.

\[33\] J.N. Mordeson, P.S. Nair, Fuzzy Graphs and Fuzzy Hypergraphs,
Physica-Verlag, Heidelberg, 2000.

\[34\] R.R. Yager, On ordered weighted averaging aggregation operators
in multicriteria decisionmaking, IEEE Trans. Syst. Man Cybern. 18 (1988)
183--190.

\[35\] E.A. Leicht, M.E.J. Newman, Community structure in directed
networks, Phys. Rev. Lett. 100 (2008) 118703.
https://doi.org/10.1103/PhysRevLett.100.118703

\[36\] M. Grabisch, Set Functions, Games and Capacities in Decision
Making, Springer, Cham, 2016.

\[37\] E.H. Ruspini, A new approach to clustering, Inf. Control 15
(1969) 22--32.

\[38\] L.A. Zadeh, Fuzzy sets as a basis for a theory of possibility,
Fuzzy Sets Syst. 1 (1978) 3--28.

\[39\] R.R. Yager, A. Rybalov, Uninorm aggregation operators, Fuzzy Sets
Syst. 80 (1996) 111--120.

\[40\] S. Pandey, G. Karypis, A self-attentive model for knowledge
tracing, in: Proc. 12th Int. Conf. Educational Data Mining, 2019,
pp. 384--389.

\[41\] A. Ghosh, N. Heffernan, A.S. Lan, Context-aware attentive
knowledge tracing, in: Proc. 26th ACM SIGKDD, 2020, pp. 2330--2339.

\[42\] J. Gawlikowski, C.R.N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng,
A. Kruspe, R. Triebel, P. Jung, R. Roscher, M. Shahzad, W. Yang, R.
Bamler, X.X. Zhu, A survey of uncertainty in deep neural networks,
Artif. Intell. Rev. 56 (2023) 1513--1589.
https://doi.org/10.1007/s10462-023-10562-9

\[43\] A.T. Corbett, J.R. Anderson, Knowledge tracing: modeling the
acquisition of procedural knowledge, User Model. User-Adapt. Interact. 4
(1994) 253--278.

\[44\] M.E.J. Newman, Analysis of weighted networks, Phys. Rev. E 70
(2004) 056131.

\[45\] J. Reichardt, S. Bornholdt, Statistical mechanics of community
detection, Phys. Rev. E 74 (2006) 016110.

\[46\] A. Arenas, A. Fernández, S. Gómez, Analysis of the structure of
complex networks at different resolution levels, New J. Phys. 10 (2008)
053039.

\[47\] V.A. Traag, P. Van Dooren, Y. Nesterov, Narrow scope for
resolution-limit-free community detection, Phys. Rev. E 84 (2011)
016114.

\[48\] A. Lancichinetti, S. Fortunato, Consensus clustering in complex
networks, Sci. Rep. 2 (2012) 336.

\[49\] M. Gondran, M. Minoux, Graphs, Dioids and Semirings: New Models
and Algorithms, Springer, New York, 2008.

\[50\] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin,
J.V. Dillon, B. Lakshminarayanan, J. Snoek, Can you trust your model's
uncertainty? Evaluating predictive uncertainty under dataset shift, in:
Advances in Neural Information Processing Systems 32, 2019, pp.
13991--14002.
