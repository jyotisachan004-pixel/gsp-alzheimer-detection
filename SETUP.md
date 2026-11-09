# SETUP

| Item | Value |
|---|---|
| Date of run | 8 November 2026 (06:19 UTC) |
| Platform | Google Colab (free tier), CPU runtime |
| OS | Ubuntu 24.04.5 LTS |
| Python | 3.13.18 |
| CPU / RAM | 2 vCPU, Intel Xeon @ 2.20 GHz; 12 GiB RAM |
| GPU | none (nvidia-smi not available; not needed) |
| Dataset | OpenNeuro ds004504, 88 subjects (59 dementia, 29 healthy), version: latest available on OpenNeuro on 8 Nov 2026 |
| Package versions | see requirements.txt (pip freeze from the Colab run) |
| Random seeds | 42, 43, 44 |
| Run time | not recorded |
| Reductions | first 1000 time bins per recording; smaller grids for kNN, Random Forest, XGBoost; TabTransformer not run |
