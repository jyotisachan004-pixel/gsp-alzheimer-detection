# NOTES - problems, attempts and fixes (do not delete failed attempts)

## A. Found by reading the authors' code BEFORE running (to be confirmed by your run)
| # | Observation | Where | What this repo does |
|---|---|---|---|
| A1 | `features_tv.csv` is not in the repo; the feature cell uses a hard-coded Windows path (`C:\Users\xmoot\...`) | `main.ipynb` cell 4 | path is a CLI argument (`scripts/02_extract_features.py`) |
| A2 | Feature rows are merged with `participants.tsv` **by row index**, but files come from an unsorted `os.scandir`. On Colab/Linux the order is arbitrary, so labels could be silently attached to the wrong subjects | cells 4 and 6 | files sorted, merged by `participant_id` (assert on row count) |
| A3 | `spectral_entropy` is an array per subject (scipy `entropy` on a matrix); in the CSV it becomes a truncated string and cannot be scaled | cell 4 | scalar from the paper's formula H(U) |
| A4 | Paper says graph nodes are the 19 electrodes; code builds the graph on **time bins** (`distance_matrix` of a time x channel table), giving ~8000 nodes and a very slow double loop for total variation | cell 4 | vectorised TV (checked identical in `tests/check_equivalence.py`); `--max-bins 1000` |
| A5 | Paper text says the SVM uses a **linear** kernel; code uses `SVC(probability=True)` = default RBF | cell 42 | kept as in code; flagged in report |
| A6 | Paper lists 11 features; binary models use 9 (`unique_clusters`, `heat_trace` excluded; saved output shows `heat_trace` = 0.0) | cell 32 | same 9 features |
| A7 | Split is not stratified; test set has only 4 healthy controls out of 27 | cells 34-44 | same split logic; HC recall and majority baseline reported |
| A8 | In the authors' saved notebook output, SVM test accuracy 0.85 comes with HC recall 0.00 and AUC 0.42 - i.e. it equals the "predict dementia for everyone" rate (23/27). `avg_degree` values look like the node count, which suggests a near-complete graph | cell 42 output | **hypotheses to test with your own numbers** (columns `hc_recall`, `mean_majority` in `my_results.csv`) |
| A9 | No LICENSE file upstream; no requirements.txt; data not bundled (OpenNeuro, version 1.0.6 in the paper) | repo root | own-files MIT licence with explicit scope; requirements added |
| A10 | The paper is a medRxiv preprint (not peer reviewed) | - | state in the report |

## B. Attempts log (fill in as you go - date, what you tried, outcome)
| Date | Step | What failed | Fix attempt | Outcome |
|---|---|---|---|---|
| TODO | | | | |

## C. Run notes
- Time taken per stage: TODO
- Anything you reduced (write exactly what): `--max-bins 1000`, reduced grids, 3 seeds, TabTransformer skipped.
