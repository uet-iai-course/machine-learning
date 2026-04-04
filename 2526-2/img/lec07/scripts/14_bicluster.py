"""Biclustering heatmap on Iris dataset."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

iris = load_iris()
X = iris.data
feature_names = ["Dài đài hoa", "Rộng đài hoa", "Dài cánh hoa", "Rộng cánh hoa"]
target = iris.target
colors_target = np.array(["#4a90d9", "#e8a020", "#5aaa44"])[target]

# Cluster observations (rows)
Z_rows = linkage(X, method="average")
row_order = dendrogram(Z_rows, no_plot=True)["leaves"]

# Cluster features (cols)
Z_cols = linkage(X.T, method="average")
col_order = dendrogram(Z_cols, no_plot=True)["leaves"]

X_ordered = X[np.array(row_order), :][:, np.array(col_order)]

fig = plt.figure(figsize=(10, 6.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 4, 0.3], height_ratios=[1.2, 4],
                       hspace=0.02, wspace=0.05)

# Top dendrogram (cols)
ax_dcol = fig.add_subplot(gs[0, 1])
dendrogram(Z_cols, ax=ax_dcol, color_threshold=0,
           above_threshold_color="#888",
           link_color_func=lambda k: "#888",
           no_labels=True)
ax_dcol.axis("off")

# Left dendrogram (rows)
ax_drow = fig.add_subplot(gs[1, 0])
dendrogram(Z_rows, ax=ax_drow, orientation="left",
           color_threshold=0, above_threshold_color="#888",
           link_color_func=lambda k: "#888",
           no_labels=True)
ax_drow.axis("off")

# Heatmap
ax_heat = fig.add_subplot(gs[1, 1])
Xn = (X_ordered - X_ordered.min(axis=0)) / (X_ordered.max(axis=0) - X_ordered.min(axis=0) + 1e-9)
im = ax_heat.imshow(Xn, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
feat_ordered = [feature_names[i] for i in col_order]
ax_heat.set_xticks(range(4))
ax_heat.set_xticklabels(feat_ordered, fontsize=10, rotation=30, ha="right")
ax_heat.set_yticks([])
ax_heat.set_title("")

# Colorbar
ax_cb = fig.add_subplot(gs[1, 2])
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=7)

fig.savefig("../bicluster-iris.svg", format="svg", bbox_inches="tight")
print("Saved bicluster-iris.svg")
