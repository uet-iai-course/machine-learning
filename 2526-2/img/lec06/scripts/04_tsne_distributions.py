"""Gaussian vs t-distribution comparison for crowding problem explanation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as tdist

x = np.linspace(-5, 5, 400)
gauss = norm.pdf(x, 0, 1)
t1 = tdist.pdf(x, df=1)  # t with 1 degree of freedom (Cauchy-like)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

# --- Left: density comparison ---
ax = axes[0]
ax.plot(x, gauss, color="#4a90d9", lw=2.2, label="Gaussian (SNE)")
ax.plot(x, t1,    color="#e8732a", lw=2.2, label="Phân phối t, df=1 (t-SNE)")
ax.fill_between(x, gauss, alpha=0.12, color="#4a90d9")
ax.fill_between(x, t1,    alpha=0.12, color="#e8732a")
ax.set_xlabel("Khoảng cách đến điểm đen", fontsize=10)
ax.set_ylabel("Độ tương đồng", fontsize=10)
ax.set_title("So sánh hàm kernel", fontsize=10)
ax.legend(fontsize=8.5, loc="upper right")
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)
ax.tick_params(labelsize=8)
ax.annotate("Đuôi nặng hơn\n→ khoảng cách trung bình\nkhông bị chật chội",
            xy=(3.5, tdist.pdf(3.5, df=1)), xytext=(2.5, 0.28),
            fontsize=7.5, color="#e8732a",
            arrowprops=dict(arrowstyle="->", color="#e8732a", lw=1.0))

# --- Right: what the crowding problem looks like ---
rng = np.random.default_rng(7)
ax2 = axes[1]
ax2.set_aspect("equal")
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_title("Vấn đề chật chội", fontsize=10)

# 3D: one center point, many equidistant neighbors in 3D -> they can't all fit in 2D
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
# In high-dim: 12 neighbors at distance 2
# In 2D: they crowd around radius ~2 but overlap
cx, cy = 0, 0
r_hd = 2.2   # high-dim radius (implied)
r_ld_ideal = 2.2
# Show Gaussian: they get pushed together (crowded)
r_gauss = 1.2
pts_gauss = np.column_stack([cx + r_gauss * np.cos(theta),
                              cy + r_gauss * np.sin(theta)])
ax2.scatter(*pts_gauss.T, s=40, color="#4a90d9", alpha=0.7, zorder=3, label="Gaussian (chật)")
ax2.scatter(cx, cy, s=80, color="black", zorder=5)

# t-SNE: more spread out
pts_t = np.column_stack([cx + r_ld_ideal * np.cos(theta),
                          cy + r_ld_ideal * np.sin(theta)])
ax2.scatter(*pts_t.T, s=40, color="#e8732a", alpha=0.7, zorder=3, marker="^",
            label="t-SNE (thoáng)")

ax2.set_xlabel("Chiều 1", fontsize=10)
ax2.set_ylabel("Chiều 2", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
ax2.tick_params(labelsize=8)
ax2.set_facecolor("#fafafa")

fig.tight_layout(pad=1.0)
out = "../tsne-gaussian-vs-tdist.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
print(f"Saved {out}")
