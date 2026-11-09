# Reproduction: GSP + ML for Alzheimer's EEG detection (Mootoo et al., 2023)

Fork of [xmootoo/gsp-alzheimer-detection](https://github.com/xmootoo/gsp-alzheimer-detection), made for Assignment 2 (Reproduce a Paper and Publish Your Code). The authors' original `main.ipynb` is kept unchanged, and their original README is at the link above.

## Overview
- **Paper:** X. S. Mootoo et al., "Detecting Alzheimer Disease in EEG Data with Machine Learning and the Graph Discrete Fourier Transform", medRxiv, 2023 (preprint, not peer reviewed). DOI: [10.1101/2023.11.01.23297940](https://doi.org/10.1101/2023.11.01.23297940)
- **Original repo:** https://github.com/xmootoo/gsp-alzheimer-detection
- **Result re-run:** Table 3, test accuracy of six classifiers for dementia (AD + FTD) vs healthy control on OpenNeuro ds004504 (88 subjects: 59 dementia, 29 healthy).
- **Outcome: partly reproduced.** Over 3 seeds, Random Forest and Logistic Regression are within 2% of the paper; kNN, XGBoost, SVM and Naive Bayes are 10 to 14% lower. The SVM matches the paper on the authors' own split (seed 42: 0.852 vs 0.85), but its accuracy equals the majority-class baseline in all three seeds, with healthy-control recall of 0.000.

## Setup
Free Google Colab CPU runtime (no GPU needed). Details in `SETUP.md`.
```bash
git clone https://github.com/jyotisachan004-pixel/gsp-alzheimer-detection
cd gsp-alzheimer-detection
pip install -r requirements.txt
```

## How to run
```bash
bash scripts/01_download_data.sh data/ds004504            # EEG data from OpenNeuro (public)
python scripts/02_extract_features.py --max-bins 1000     # features -> results/features_tv.csv
python scripts/03_run_models.py --seeds 42 43 44          # -> results/my_results.csv
python make_figure.py                                     # -> fig1.pdf
```
Or open `run_on_colab.ipynb` in Colab and run the cells from the top. Console output is saved in `logs/`.

## Expected output
- `results/my_results.csv`: one row per model with the paper value, mean and SD over 3 seeds, % difference, healthy-control recall and majority baseline.
- `results/per_seed.csv`: every model and seed.
- `fig1.pdf`: paper vs reproduced accuracy.

## Differences from the paper
| Item | Paper / original code | This reproduction |
|---|---|---|
| Seeds | one split | 3 seeds (42, 43, 44); seed 42 is the authors' split |
| Recording length | whole recording (about 8000 time bins) | first 1000 time bins |
| Hyper-parameter grids (kNN, Random Forest, XGBoost) | very large grids | small grids (see `scripts/03_run_models.py`) |
| TabTransformer | included | not run |
| Spectral entropy | array stored in CSV | scalar from the paper's formula |
| Merge with `participants.tsv` | by row index | by `participant_id` |

All problems found and fixes are in `NOTES.md`.

## Citation
See `CITATION.cff`. Please also cite the original paper and the dataset (Miltiadous et al., Data 8(6):95, 2023).

## Licence
See `LICENSE`. It covers only the files added in this fork. The original repository had no licence file.
