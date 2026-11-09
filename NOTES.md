# NOTES: problems, attempts and fixes

## A. Found by reading the authors' code before running
| # | Observation | What this repo does |
|---|---|---|
| A1 | The feature file `features_tv.csv` is not in the original repo, and the feature cell uses a hard-coded Windows path. | Path is a command-line argument in `scripts/02_extract_features.py`. |
| A2 | Features are merged with `participants.tsv` by row index after an unsorted directory listing, which can attach labels to the wrong subjects on Linux. (Not tested on the original code.) | Files are sorted and merged by `participant_id`, with a row-count check. |
| A3 | `spectral_entropy` is stored as an array and cannot be reloaded from CSV as a number. | Scalar value from the paper's formula. |
| A4 | The paper says graph nodes are the 19 electrodes, but the code builds the graph on time bins (about 8000 nodes), which is too slow for free Colab. | Total variation vectorised (checked identical in `tests/check_equivalence.py`: loop 2.254043 vs vectorised 2.254043). Only the first 1000 time bins are used. |
| A5 | The paper says a linear-kernel SVM, but the code uses the default RBF kernel. | Kept as in the code. |
| A6 | The paper lists 11 features, but the binary models in the code use 9. | Same 9 features. |
| A7 | The split is not stratified, so the number of healthy controls in the test set changes a lot. | Same split logic; healthy-control recall and the majority baseline are also reported. |
| A8 | No licence and no requirements file in the original repo; the data is not bundled; the paper is a preprint. | Own licence for added files only; requirements added; data downloaded from OpenNeuro. |

## B. Attempts log
| Step | What went wrong | Fix attempted | Outcome |
|---|---|---|---|
| Figure | In the first figure, the x-axis model names overlapped and "Logistic Regression" was cut off. | Rotated the labels by 30 degrees and moved the legend lower in `make_figure.py`. | First edit caused a SyntaxError on the legend line (a typo: `(0.5, -0.30 to -0.55)`). Fixed to `(0.5, -0.55)`. The figure then rendered correctly and stayed readable in grayscale. |

## C. What the run showed (own run, 8 Nov 2026, 3 seeds: 42, 43, 44)
- The SVM's test accuracy equals the majority-class baseline in all three seeds (0.852, 0.704, 0.630), with healthy-control recall 0.000 each time and a mean AUC of 0.455.
- Seed 42 is the authors' split. The SVM matches the paper there (0.852 vs 0.85), but the test set has only 4 healthy controls out of 27.
- Mean accuracy over 3 seeds is within 2% of the paper for Random Forest and Logistic Regression only; the other four models are 10 to 14% lower.
- Outcome: partly reproduced.

## D. Reductions
First 1000 time bins per recording; smaller hyper-parameter grids for kNN, Random Forest and XGBoost; 3 seeds; TabTransformer not run. Run times were not recorded.

## E. Fresh-clone test (9 Nov 2026)
- Problem: following the README on a new Colab notebook, `pip install -r requirements.txt` failed with an OSError for `google_colab-1.0.0.tar.gz`. The file was a full `pip freeze` from my first Colab session, which includes Colab's own packages, so it cannot be installed on a fresh runtime.
- Fix: renamed the full list to `requirements-freeze.txt` (kept as a record of the exact versions) and added a short installable `requirements.txt` (numpy, pandas, scipy, scikit-learn, xgboost, mne, pygsp, matplotlib, awscli).
- Outcome: after the fix, the install finished (pip printed warnings about Colab's preinstalled packages, not about this project). Download, feature extraction, models and figure ran from the README alone, and the results matched the first run (SVM mean accuracy 0.728, Random Forest 0.679, majority baseline 0.728).
