"""Draw Fig. 1: paper vs. reproduced test accuracy (mean +/- SD over seeds). Reads results/my_results.csv.
Usage: python make_figure.py            -> fig1.pdf (vector, 3.5 in wide)
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/my_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig1.pdf"
df = pd.read_csv(CSV)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "hatch.linewidth": 0.6})
# Okabe-Ito colour-blind-safe pair; hatching + different luminance keep it readable in grayscale
C_PAPER, C_MINE = "#E69F00", "#0072B2"

fig, ax = plt.subplots(figsize=(3.5, 2.6))
x = np.arange(len(df)); w = 0.38
ax.bar(x - w/2, df["paper_acc"], w, label="Paper (Table 3)", color=C_PAPER, edgecolor="black", hatch="///", linewidth=0.5)
ax.bar(x + w/2, df["mean_acc"], w, yerr=df["std_acc"].fillna(0), capsize=2, label=f"Reproduced (mean $\\pm$ SD, {int(df['n_seeds'].max())} seeds)",
       color=C_MINE, edgecolor="black", hatch="...", linewidth=0.5, error_kw={"elinewidth": 0.8})
if "mean_majority" in df:
    ax.axhline(df["mean_majority"].mean(), color="black", linestyle="--", linewidth=0.8, label="Majority-class baseline")
ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n") for m in df["model"]])
ax.set_ylabel("Test accuracy (fraction)"); ax.set_ylim(0, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False)
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
