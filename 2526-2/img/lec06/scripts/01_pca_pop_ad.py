"""PCA on simulated Population vs Advertising data — PC1/PC2 arrows."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
n = 100
# Simulate correlated data along one main direction
t = rng.uniform(10, 70, n)
pop = t + rng.normal(0, 3, n)
ad = 0.5 * t + 5 + rng.normal(0, 2, n)

X = np.column_stack([pop, ad])
X_c = X - X.mean(axis=0)

pca = PCA(n_components=2)
pca.fit(X_c)
scores = pca.transform(X_c)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# --- Left: original space with PC arrows ---
ax = axes[0]
ax.scatter(pop, ad, s=18, alpha=0.55, color="#4a90d9", zorder=3)
cx, cy = X.mean(axis=0)
scale = 22
for i, (color, ls, lbl) in enumerate(
    [("#e8732a", "-", "PC1"), ("#555555", "--", "PC2")]
):
    dx, dy = pca.components_[i] * scale * (1 if i == 0 else 0.45)
    ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=2.2 if i == 0 else 1.4,
                                linestyle="solid" if i == 0 else "dashed"))
    ax.text(cx + dx * 1.05, cy + dy * 1.05, lbl, fontsize=9,
            color=color, ha="center", va="bottom")
ax.set_xlabel("Dân số", fontsize=10)
ax.set_ylabel("Quảng cáo", fontsize=10)
ax.set_title("Không gian gốc", fontsize=10)
ax.tick_params(labelsize=8)

# --- Right: PC score space ---
ax2 = axes[1]
ax2.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.55, color="#4a90d9", zorder=3)
ax2.axhline(0, color="#e8732a", lw=1.6, label="PC1")
ax2.axvline(0, color="#555555", lw=1.2, ls="--", label="PC2")
ax2.set_xlabel("Điểm PC1", fontsize=10)
ax2.set_ylabel("Điểm PC2", fontsize=10)
ax2.set_title("Không gian PC", fontsize=10)
ax2.tick_params(labelsize=8)

pve = pca.explained_variance_ratio_
ax2.set_title(
    f"Không gian PC  (PVE: PC1={pve[0]:.0%}, PC2={pve[1]:.0%})", fontsize=9
)

fig.tight_layout(pad=1.0)
out = "../pca-pop-ad-scatter.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print(f"Saved {out}")
