"""PCA scree / PVE plot on USArrests data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

X = StandardScaler().fit_transform(data)
pca = PCA().fit(X)
pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)
pcs = np.arange(1, len(pve) + 1)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

# --- Left: individual PVE ---
ax = axes[0]
ax.plot(pcs, pve, "o-", color="#4a90d9", lw=2, ms=7, zorder=3)
ax.set_xlabel("Thành phần chính", fontsize=10)
ax.set_ylabel("Tỷ lệ phương sai (PVE)", fontsize=10)
ax.set_title("PVE từng thành phần", fontsize=10)
ax.set_xticks(pcs)
ax.set_xticklabels([f"PC{i}" for i in pcs], fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.tick_params(labelsize=8)
ax.grid(axis="y", alpha=0.3)
# annotate elbow
ax.annotate("Điểm khuỷu", xy=(2, pve[1]), xytext=(2.8, pve[1] + 0.05),
            fontsize=8, color="#e8732a",
            arrowprops=dict(arrowstyle="->", color="#e8732a", lw=1.2))

# --- Right: cumulative PVE ---
ax2 = axes[1]
ax2.plot(pcs, cum_pve, "s-", color="#e8732a", lw=2, ms=7, zorder=3)
ax2.axhline(0.9, color="#5aaa44", lw=1.2, ls="--", label="90%")
ax2.legend(fontsize=8)
ax2.set_xlabel("Số thành phần chính", fontsize=10)
ax2.set_ylabel("PVE tích lũy", fontsize=10)
ax2.set_title("PVE tích lũy", fontsize=10)
ax2.set_xticks(pcs)
ax2.set_xticklabels([f"{i}" for i in pcs], fontsize=9)
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax2.set_ylim(0, 1.05)
ax2.tick_params(labelsize=8)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout(pad=1.0)
out = "../pca-scree-plot.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print(f"Saved {out}")
