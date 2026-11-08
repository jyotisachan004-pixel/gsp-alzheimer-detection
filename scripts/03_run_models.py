"""Step 2: reproduce Table 3 of Mootoo et al. (2023): test accuracy, binary Dementia (AD+FTD) vs HC.

Follows the authors cells 32-44 of main.ipynb: 9 features, non-stratified 70/30 split then 50/50 of the remainder
(35/35/30), RobustScaler fitted on train, same classifiers. Differences (also in NOTES.md / README):
  * 3 seeds (42, 43, 44) instead of one; seed 42 = the authors split. Seed is used for split AND model random_state.
  * GridSearch grids for kNN / RF / XGBoost are REDUCED so the run fits free Colab (original grids = 8,640 / 72,000 fits).
  * TabTransformer (7th row of the paper table) is NOT run (heavy dependency, short time budget).
  * Extra diagnostics the paper does not report: HC recall, balanced accuracy, majority-class baseline.
Outputs: results/per_seed.csv and results/my_results.csv (one row per model, used by make_figure.py).
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

FEATURES = ["stationary_ratio","Tik-norm","Total_Variation","graph_energy","spectral_entropy",
            "signal_energy","signal_power","avg_degree","diffusion_distance"]
PAPER_TABLE3 = {"kNN": 0.70, "Random Forest": 0.67, "XGBoost": 0.74, "Logistic Regression": 0.63,
                "SVM": 0.85, "Naive Bayes": 0.59}

def make_models(seed, fast):
    knn_grid = {"n_neighbors": [3,5,7,9,11,13], "weights": ["uniform","distance"], "metric": ["euclidean","manhattan"]}
    rf_grid = {"n_estimators": [100,300], "max_depth": [None,10], "min_samples_leaf": [1,4]}
    xgb_grid = {"learning_rate": [0.01,0.1], "n_estimators": [100,300], "max_depth": [3,7], "subsample": [0.8,1.0]}
    gs = lambda est, grid: GridSearchCV(est, grid, scoring="roc_auc", cv=5, n_jobs=-1)
    return {
        "kNN": gs(KNeighborsClassifier(), knn_grid),
        "Random Forest": gs(RandomForestClassifier(random_state=seed), rf_grid),
        "XGBoost": gs(XGBClassifier(random_state=seed, eval_metric="logloss"), xgb_grid),
        "Logistic Regression": LogisticRegression(random_state=seed),
        "SVM": SVC(random_state=seed, probability=True),      # code uses default RBF kernel; paper text says linear
        "Naive Bayes": GaussianNB(),
    }

def main(a):
    feats = pd.read_csv(a.features)
    parts = pd.read_csv(a.participants, sep="\t")
    data = feats.merge(parts, on="participant_id", how="inner", validate="one_to_one")   # merge BY ID (see NOTES.md)
    assert len(data) == len(feats), "ID merge lost rows - check participant_id format"
    y = (data["Group"].isin(["A","F"])).astype(int).values      # 1 = dementia (AD+FTD), 0 = healthy control
    X = data[FEATURES].values
    print("n =", len(data), "| dementia:", int(y.sum()), "| HC:", int((1-y).sum()))
    rows = []
    for seed in a.seeds:
        Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.5, random_state=seed)
        sc = RobustScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        majority = max(yte.mean(), 1 - yte.mean())
        for name, model in make_models(seed, a.fast).items():
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xte_s)
            proba = model.predict_proba(Xte_s)[:, 1]
            rows.append(dict(seed=seed, model=name, n_test=len(yte), n_hc_test=int((yte==0).sum()),
                             accuracy=accuracy_score(yte, pred), balanced_accuracy=balanced_accuracy_score(yte, pred),
                             hc_recall=recall_score(yte, pred, pos_label=0, zero_division=0),
                             dementia_recall=recall_score(yte, pred, pos_label=1, zero_division=0),
                             auc=roc_auc_score(yte, proba) if len(set(yte)) == 2 else np.nan,
                             majority_baseline=majority))
            r = rows[-1]
            print(f"seed {seed} | {name:20s} acc={r['accuracy']:.3f} HC-recall={r['hc_recall']:.2f}", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(a.per_seed, index=False)
    g = per.groupby("model", sort=False)
    out = g.agg(mean_acc=("accuracy","mean"), std_acc=("accuracy","std"), mean_bal_acc=("balanced_accuracy","mean"),
                mean_hc_recall=("hc_recall","mean"), mean_auc=("auc","mean"), mean_majority=("majority_baseline","mean"),
                n_seeds=("seed","nunique")).reset_index()
    out.insert(1, "paper_acc", out["model"].map(PAPER_TABLE3))
    out["diff_pct"] = (out["mean_acc"] - out["paper_acc"]) / out["paper_acc"] * 100
    out["within_2pct"] = out["diff_pct"].abs() <= 2
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features_tv.csv")
    ap.add_argument("--participants", default="data/ds004504/participants.tsv")
    ap.add_argument("--per-seed", dest="per_seed", default="results/per_seed.csv")
    ap.add_argument("--out", default="results/my_results.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--fast", action="store_true")
    main(ap.parse_args())
