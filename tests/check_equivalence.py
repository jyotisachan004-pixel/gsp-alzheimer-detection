"""Sanity check: vectorised total variation == authors' double loop (random data, no EEG needed)."""
import sys, os
import numpy as np
from scipy.spatial import distance_matrix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("fe", os.path.join(os.path.dirname(__file__), "..", "scripts", "02_extract_features.py"))
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)

def tv_loop(W, X):                       # authors' original implementation
    tv = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            tv += W[i, j] * np.linalg.norm(X[j] - X[i]) ** 2
    return np.sqrt(tv)

rng = np.random.default_rng(0)
X = rng.normal(scale=0.3, size=(60, 19))
D = distance_matrix(X, X); W = np.exp(-D**2); W[D > 1] = 0; np.fill_diagonal(W, 0)
a, b = tv_loop(W, X), fe.total_variation(W, D)
print("loop:", a, "vectorised:", b)
assert np.isclose(a, b), "MISMATCH"
print("OK: identical")
