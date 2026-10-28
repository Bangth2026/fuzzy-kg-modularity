# Data

No dataset is redistributed in this repository. Every corpus used in the paper
is public; download the files below into `data/raw/` and the construction
scripts will find them.

| Corpus | What the paper uses | Where to get it | Expected path |
|---|---|---|---|
| ASSISTments 2009–2010 skill-builder | `skill_builder_data.csv` (or the `_corrected` variant) | ASSISTments Data Portal; mirrored on figshare, article 25309000 | `data/raw/assist2009/skill_builder_data.csv` |
| ASSISTments 2015 | `2015_100_skill_builders_main_problems.csv` | ASSISTments Data Portal; EduData mirror `base.ustc.edu.cn/data/ASSISTment/` | `data/raw/assist2015/2015_100_skill_builders_main_problems.csv` |
| ASSISTments 2017 | `anonymized_full_release_competition_dataset.csv` | ASSISTments Data Portal; EduData mirror | `data/raw/assist2017/anonymized_full_release_competition_dataset.csv` |
| MOOCCubeX | `prerequisites/cs.json`, `relations/concept-problem.txt` | https://github.com/THU-KEG/MOOCCubeX | `data/raw/MOOCCubeX/prerequisites/cs.json`, `data/raw/MOOCCubeX/relations/concept-problem.txt` |
| Junyi Academy (Jan 2015 release; non-commercial use, cite Chang, Hsu and Chen, EDM 2015) | `junyi_ProblemLog_original.csv`, `junyi_Exercise_table.csv`, `relationship_annotation_{training,testing}.csv` | PSLC DataShop, dataset 1198: https://pslcdatashop.web.cmu.edu/Files?datasetId=1198 | `data/raw/junyi/junyi/` |

The 22.4 GB MOOCCubeX interaction log is **not** required: the MOOCCubeX-CS
graph of Table 2 is built from the two small files above (the released
prerequisite confidences are used directly as `mu_E`).

## Preprocessing (Section 8, stated in the paper and implemented in `loaders.py`)

* rows with a missing skill are dropped;
* scaffolding rows are removed where the release marks them (`original = 1` kept, 2009 and 2017);
* duplicate log entries are removed;
* responses outside {0, 1} are dropped (2015);
* sequences are ordered by the release's log order (`time_done` on Junyi);
* on Junyi a response is the release's `correct` flag (first attempt correct and no hint requested);
* learners with fewer than 15 retained interactions are excluded.

KC-level nodes are skills (2009, 2017) or skill-builder problem sets (2015);
item-level nodes are individual problems (2009, 2017); Junyi nodes are exercises.

The ASSISTments 2009 release has known undisclosed preprocessing upstream, so
reconstructions are reproducible but not guaranteed identical to the counts in
the published counts to the last digit.

## Output

`scripts/build_assistments_graph.py`, `scripts/build_junyi_graph.py` and `scripts/build_mooccubex_graph.py`
write `data/graphs/<name>_edges.csv` (columns `src,dst,mu` with integer node
ids), `<name>_nodes.csv` (id → label) and `<name>_signals.csv` (the three
signals per retained pair). Every downstream script reads those CSVs.
