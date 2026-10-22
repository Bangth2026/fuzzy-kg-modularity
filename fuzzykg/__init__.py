"""fuzzykg: reference implementation for

    Truong, H. B. "Modularity, interval type-2 states and weakest-link
    composition on fuzzy knowledge graphs." Fuzzy Sets and Systems (submitted).

Modules
-------
graph          FuzzyGraph container (directed, memberships in (0,1]).
modularity     Q_F (Eq. 7-8), alpha-cut modularity (Eq. 9), Proposition 3 /
               Corollary 1 numerical verification, Remark 3 threshold.
leiden         Fuzzy Leiden (Leiden with Q_F as objective), Proposition 1 and
               Proposition 2(iii) checks.
resolution     Algorithm 1 (adaptive resolution from the normalised cut).
fuzzification  Eq. (10)-(13): type-1 collapse, footprint of uncertainty,
               Nie-Tan reduction, three-valued ZPD gate, h* of Section 5.3.
tnorms         min / product / Lukasiewicz, Yager and Schweizer-Sklar
               families, OWA, Fact 1 check, max-min widest path (Eq. 1).
construction   Eq. (14)-(15): fuzzy knowledge graph from interaction logs.
synthetic      Synthetic fuzzy prerequisite graphs of Section 8.2.
"""

from .graph import FuzzyGraph
from .modularity import (
    fuzzy_modularity,
    alpha_cut_modularity,
    alpha_average_modularity,
    correlation_gap,
    bipartition_threshold,
)
from .leiden import fuzzy_leiden, check_weak_connectivity, check_monotone
from .resolution import adaptive_resolution
from .fuzzification import (
    collapse_type1,
    footprint,
    nie_tan,
    zpd_gate_it2,
    zpd_gate_type1,
    degeneration_halfwidth,
)
from .tnorms import (
    t_min,
    t_product,
    t_lukasiewicz,
    yager,
    schweizer_sklar,
    owa,
    owa_weights_from_orness,
    orness,
    compose_path,
    widest_path,
)

__version__ = "1.0.0"
__all__ = [
    "FuzzyGraph",
    "fuzzy_modularity",
    "alpha_cut_modularity",
    "alpha_average_modularity",
    "correlation_gap",
    "bipartition_threshold",
    "fuzzy_leiden",
    "check_weak_connectivity",
    "check_monotone",
    "adaptive_resolution",
    "collapse_type1",
    "footprint",
    "nie_tan",
    "zpd_gate_it2",
    "zpd_gate_type1",
    "degeneration_halfwidth",
    "t_min",
    "t_product",
    "t_lukasiewicz",
    "yager",
    "schweizer_sklar",
    "owa",
    "owa_weights_from_orness",
    "orness",
    "compose_path",
    "widest_path",
]
