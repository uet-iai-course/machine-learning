"""PCA as coordinate transformation — 3D data → 2D principal-component plane.

Side-by-side:
  Trái: dữ liệu 3D (X1, X2, X3), nằm gần một mặt phẳng 2D có hướng nghiêng;
        ba trục PC1, PC2, PC3 vẽ tại gốc với độ dài ∝ √λ.
  Phải: chiếu lên (Z1, Z2) — mặt phẳng chính. Cùng điểm giữ màu để mắt
        theo dõi việc biến đổi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

rng = np.random.default_rng(11)
n = 100

# Dữ liệu nằm gần một mặt phẳng nghiêng trong không gian 3D:
# σ lớn theo hai hướng đầu, σ nhỏ theo hướng thứ ba (gần phẳng)
data = np.column_stack([
    rng.normal(0, 2.4, n),    # phương sai lớn nhất
    rng.normal(0, 1.4, n),    # phương sai trung
    rng.normal(0, 0.32, n),   # phương sai nhỏ — chiều "thừa"
])

# Xoay sang một hướng tuỳ ý để mặt phẳng chính nghiêng trong (X1,X2,X3)
# Quaternion-style: dùng 2 góc Euler đơn giản
def Rz(t): return np.array([[np.cos(t), -np.sin(t), 0],
                            [np.sin(t),  np.cos(t), 0],
                            [0, 0, 1]])
def Ry(t): return np.array([[ np.cos(t), 0, np.sin(t)],
                            [ 0, 1, 0],
                            [-np.sin(t), 0, np.cos(t)]])
R = Rz(np.deg2rad(35)) @ Ry(np.deg2rad(28))
X = data @ R.T
X_c = X - X.mean(axis=0)

pca = PCA(n_components=3)
pca.fit(X_c)
scores = pca.transform(X_c)
v1, v2, v3 = pca.components_
lam1, lam2, lam3 = pca.explained_variance_

# Identity color: gradient theo PC1 score
ranks = np.argsort(np.argsort(scores[:, 0]))
colors = plt.cm.viridis(ranks / (n - 1))

fig = plt.figure(figsize=(11.5, 4.6))

# Symmetric limits (cùng phạm vi cho đẹp)
m3 = np.abs(X_c).max() * 1.15
m2 = np.abs(scores[:, :2]).max() * 1.18

arr_scale = 1.4
L1 = np.sqrt(lam1) * arr_scale
L2 = np.sqrt(lam2) * arr_scale
L3 = np.sqrt(lam3) * arr_scale

# ── Left: 3D scatter + PC arrows ──
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(X_c[:, 0], X_c[:, 1], X_c[:, 2],
           c=colors, s=28, alpha=0.92,
           edgecolors="white", linewidths=0.4, depthshade=False)

# PC arrows from origin (use quiver for 3D arrows)
for vec, length, color, label, lab_offset in [
    (v1, L1, "#2c6ea3", "PC1", 1.18),
    (v2, L2, "#e8732a", "PC2", 1.55),
    (v3, L3, "#888888", "PC3", 1.9),
]:
    end = vec * length
    ax.quiver(0, 0, 0, end[0], end[1], end[2],
              color=color, arrow_length_ratio=0.18, linewidth=2.4)
    p = vec * length * lab_offset
    ax.text(p[0], p[1], p[2], label, color=color,
            fontsize=11.5, fontweight="bold")

ax.set_xlabel(r"$X_1$", labelpad=-8, fontsize=10)
ax.set_ylabel(r"$X_2$", labelpad=-8, fontsize=10)
ax.set_zlabel(r"$X_3$", labelpad=-8, fontsize=10)
ax.set_title("Không gian gốc 3D — PC chéo với trục", fontsize=11.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(-m3, m3)
ax.set_ylim(-m3, m3)
ax.set_zlim(-m3, m3)
ax.view_init(elev=18, azim=-58)

# ── Right: 2D projection on PC1-PC2 plane ──
ax = fig.add_subplot(1, 2, 2)
ax.axhline(0, color="#2c6ea3", lw=1.4, alpha=0.7, zorder=0)
ax.axvline(0, color="#e8732a", lw=1.4, alpha=0.7, zorder=0)
ax.scatter(scores[:, 0], scores[:, 1], c=colors, s=42,
           alpha=0.92, edgecolors="white", linewidths=0.5, zorder=2)

ax.annotate("", xy=(L1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#2c6ea3",
                            lw=2.6, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(0, L2), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#e8732a",
                            lw=2.6, shrinkA=0, shrinkB=0))
ax.text(L1 * 1.06, 0, "PC1", color="#2c6ea3", fontsize=13,
        fontweight="bold", ha="left", va="center")
ax.text(0, L2 * 1.55, "PC2", color="#e8732a", fontsize=13,
        fontweight="bold", ha="center", va="bottom")

ax.set_xlabel(r"$Z_1$ (PC1 score)", fontsize=11)
ax.set_ylabel(r"$Z_2$ (PC2 score)", fontsize=11)
ax.set_title("Chiếu xuống mặt phẳng chính — 2D", fontsize=11.5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-m2, m2)
ax.set_ylim(-m2, m2)

# Tỉ lệ phương sai giữ lại
total = lam1 + lam2 + lam3
kept = (lam1 + lam2) / total * 100
ax.text(0.5, -0.13,
        f"Phương sai giữ lại: $\\lambda_1 + \\lambda_2 = {kept:.1f}\\%$ tổng",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#333")

# 3D + tight_layout không hợp — dùng subplots_adjust + pad_inches
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08, wspace=0.05)
fig.savefig("../pca-as-rotation.svg", format="svg",
            bbox_inches="tight", pad_inches=0.3)
print("Saved pca-as-rotation.svg")
