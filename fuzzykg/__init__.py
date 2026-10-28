"""fuzzykg: reference implementation for

    Truong, H. B. "Level-set modularity for fuzzy knowledge graphs: pricing
    binarisation and non-idempotent t-norms against planted partitions, null
    models and held-out transitions".

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
synthetic      Synthetic fuzzy prerequisite graphs, planted partitions with
               controllable membership information, LFR-style directed benchmark.
multislice     Level-set (multislice) modularity over the alpha-cut family, the
               binarisation cost index and the persistence diagnostic.
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

__version__ = "1.4.1"
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

# additions in 1.1.0
from .tnorms import generator, generator_best_paths  # noqa: E402
from .synthetic import planted_partition_fuzzy_graph  # noqa: E402
from .resolution import null_model_zscore, adaptive_resolution_with_null  # noqa: E402
from .modularity import bipartition_threshold_general, community_vs_rest_candidates  # noqa: E402
__all__ += ["generator", "generator_best_paths", "planted_partition_fuzzy_graph",
            "null_model_zscore", "adaptive_resolution_with_null",
            "bipartition_threshold_general", "community_vs_rest_candidates"]

# additions in 1.2.0
from .multislice import multislice_leiden, level_grid, qbar, binarisation_cost, persistence, persistence_cost  # noqa: E402
from .synthetic import lfr_style_fuzzy_graph  # noqa: E402
__all__ += ["multislice_leiden", "level_grid", "qbar", "binarisation_cost", "persistence", "persistence_cost",
            "lfr_style_fuzzy_graph"]
