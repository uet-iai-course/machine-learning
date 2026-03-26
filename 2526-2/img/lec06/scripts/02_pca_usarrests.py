"""PCA biplot on USArrests-equivalent data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# USArrests data (50 states, 4 variables)
# Source: R datasets — exact values
states = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire",
    "New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio",
    "Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota",
    "Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
    "Wisconsin","Wyoming"
]
# Murder, Assault, UrbanPop, Rape
data = np.array([
    [13.2,236,58,21.2],[10.0,263,48,44.5],[8.1,294,80,31.0],[8.8,190,50,19.5],
    [9.0,276,91,40.6],[7.9,204,78,38.7],[3.3,110,77,11.1],[5.9,238,72,15.8],
    [15.4,335,80,31.9],[17.4,211,60,25.8],[5.3,46,83,20.2],[2.6,120,54,14.2],
    [10.4,249,83,24.0],[7.2,113,65,21.0],[2.2,56,57,11.3],[6.0,115,66,18.0],
    [9.7,109,52,16.3],[15.4,249,66,22.2],[2.1,83,51,7.8],[11.3,300,67,27.8],
    [4.4,149,85,16.3],[12.1,255,74,35.1],[2.7,72,66,14.9],[16.1,259,44,17.1],
    [9.0,178,70,28.2],[6.0,109,53,16.4],[4.3,102,62,16.5],[12.2,252,81,46.0],
    [2.1,57,56,9.5],[7.4,159,89,18.8],[11.4,285,70,32.1],[11.1,254,86,26.1],
    [13.0,337,45,16.1],[0.8,45,44,7.3],[7.3,120,75,21.4],[6.6,151,68,20.0],
    [4.9,159,67,29.3],[6.3,106,72,14.9],[3.4,174,87,8.3],[14.4,279,48,22.5],
    [3.8,86,45,12.8],[13.2,188,59,26.9],[12.7,201,80,25.5],[3.2,120,80,22.9],
    [2.2,48,32,11.2],[8.5,156,63,20.7],[4.0,145,73,26.2],[5.7,81,39,9.3],
    [2.6,53,66,10.8],[6.8,161,60,15.6]
])

features = ["Murder", "Assault", "UrbanPop", "Rape"]
X = StandardScaler().fit_transform(data)
pca = PCA(n_components=2)
scores = pca.fit_transform(X)
loadings = pca.components_.T  # shape (4, 2)

fig, ax = plt.subplots(figsize=(7, 5.5))

# State points
ax.scatter(scores[:, 0], scores[:, 1], s=14, color="#4a90d9", alpha=0.7, zorder=3)
for i, s in enumerate(states):
    ax.text(scores[i, 0] + 0.04, scores[i, 1] + 0.04, s,
            fontsize=5.5, color="#333", alpha=0.85)

# Loading arrows
scale = 3.2
colors = ["#e8732a", "#c0392b", "#5aaa44", "#8e44ad"]
for j, (feat, col) in enumerate(zip(features, colors)):
    dx, dy = loadings[j, 0] * scale, loadings[j, 1] * scale
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                mutation_scale=10))
    ax.text(dx * 1.08, dy * 1.08, feat, fontsize=9, color=col,
            fontweight="bold", ha="center", va="center")

ax.axhline(0, color="#ccc", lw=0.8, zorder=1)
ax.axvline(0, color="#ccc", lw=0.8, zorder=1)
pve = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1  ({pve[0]:.1%} phương sai)", fontsize=10)
ax.set_ylabel(f"PC2  ({pve[1]:.1%} phương sai)", fontsize=10)
ax.set_title("Biplot PCA — Dữ liệu USArrests", fontsize=11)
ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.8)
out = "../pca-usarrests-biplot.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print(f"Saved {out}")
