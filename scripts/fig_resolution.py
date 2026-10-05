"""Fig. 3: test accuracy as a function of resolution, from results/results.csv."""

import argparse
import csv

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", default="results/results.csv")
parser.add_argument("--out", default="results/ResolutionFigure.pdf")
args = parser.parse_args()

with open(args.results) as f:
    rows = sorted(csv.DictReader(f), key=lambda r: -int(r["resolution_m"]))
labels = [r["resolution_m"] for r in rows]
accuracy = [float(r["accuracy"]) for r in rows]
x = list(range(1, len(rows) + 1))

fig, ax = plt.subplots()
ax.set_ylim(0.8, 1.0)
ax.set_xlabel("Resolution (m)")
ax.set_ylabel("Accuracy")
ax.set_aspect(18)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.plot(x, accuracy)
ax.plot(x, accuracy, "o")
if "2" in labels:
    ax.annotate(
        "Single trees become distinguishable",
        xy=(labels.index("2") + 1, 0.95),
        xytext=(len(x), 0.860),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        bbox=dict(facecolor="white", edgecolor="none"),
        ha="right",
        va="center",
    )
ax.grid()
fig.savefig(args.out, bbox_inches="tight")
