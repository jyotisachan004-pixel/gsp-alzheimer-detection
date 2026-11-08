"""Step 1 of the authors pipeline: GSP features per recording.

Port of cell 4 of the authors main.ipynb (xmootoo/gsp-alzheimer-detection). Changes vs the original (all logged in NOTES.md):
  * hard-coded Windows path -> command-line argument;
  * files processed in SORTED order and participant_id stored, so rows merge with participants.tsv by ID
    (original merged by row index after an unsorted os.scandir);
  * spectral_clustering random_state fixed (original: default);
  * spectral_entropy computed as the scalar defined in the paper (original stored an array, which breaks on CSV round-trip);
  * total variation computed in vectorised form sqrt(sum(W * D^2)), which is algebraically identical to the
    original double loop (checked in tests/check_equivalence.py) but ~1000x faster;
  * --max-bins: use only the first N median-binned time steps per recording (REDUCTION so it fits free Colab).
    In the authors code the graph nodes are the time bins (not the 19 electrodes as the paper text says), so cost grows as N^3.
"""
import argparse, os, glob
import numpy as np, pandas as pd, mne
from pygsp import graphs
from scipy.spatial import distance_matrix
from sklearn.cluster import spectral_clustering

CHANNELS = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
THETA, K = 1.0, 1.0

def total_variation(W, D):
    return float(np.sqrt(np.sum(W * D ** 2)))

def spectral_entropy(U):
    """Paper Eq.: H(U) = -sum_ij U_ij^2 log(U_ij^2) (scalar). The authors code called scipy entropy(U**2), which returns a
    per-column ARRAY; written to CSV it becomes a truncated string that sklearn cannot read (see NOTES.md)."""
    u2 = np.square(U)
    return float(-np.sum(u2[u2 > 0] * np.log(u2[u2 > 0])))

def gsp_features(df, seed=0):
    X = df.values
    D = distance_matrix(X, X)
    W = np.exp(-D ** 2 / THETA ** 2)
    W[D > K] = 0
    np.fill_diagonal(W, 0)
    G = graphs.Graph(W)
    L = G.L.toarray()
    ev, U = np.linalg.eigh(L)
    Xg = U.T @ X
    C = np.cov(Xg)
    T = U.T.conj() @ C @ U
    r = np.linalg.norm(np.diag(T)) / np.linalg.norm(T, "fro")
    P = L @ X
    Y = np.sum(X * P) ** 2
    labels = spectral_clustering(W, random_state=seed)
    return {
        "stationary_ratio": r, "Tik-norm": Y, "Total_Variation": total_variation(W, D),
        "graph_energy": np.sum(np.abs(ev)), "spectral_entropy": spectral_entropy(U),
        "signal_energy": np.sum(np.square(X)), "signal_power": np.var(X),
        "unique_clusters": len(np.unique(labels)), "avg_degree": np.mean(np.sum(W, axis=0)),
        "heat_trace": np.trace(np.exp(-L)), "diffusion_distance": np.sum(np.exp(-L)),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ds004504")
    ap.add_argument("--out", default="results/features_tv.csv")
    ap.add_argument("--max-bins", type=int, default=1000, help="time bins per recording (REDUCTION; original = all, ~8000)")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N recordings")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "derivatives", "*", "eeg", "*.set")))
    if a.limit: files = files[:a.limit]
    if not files: raise SystemExit(f"No .set files under {a.data}/derivatives")
    print(f"Found {len(files)} .set files; max_bins={a.max_bins}", flush=True)
    rows = {}
    for n, f in enumerate(files, 1):
        sub = os.path.basename(os.path.dirname(os.path.dirname(f)))
        raw = mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR")
        data = pd.DataFrame(raw.get_data(picks=CHANNELS).T, columns=CHANNELS)
        data = data.groupby(data.index // 50).median().iloc[: a.max_bins]
        rows[sub] = gsp_features(data)
        print(f"[{n}/{len(files)}] {sub} done ({len(data)} bins)", flush=True)
        pd.DataFrame(rows).T.to_csv(a.out, index_label="participant_id")
    print("Saved", a.out)
