"""Draw the simple Hitters decision tree (Years < 4.5, Hits < 117.5) with simulated data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
n = 260
years = np.random.exponential(5, n).clip(1, 24).astype(int)
hits = np.random.normal(100, 50, n).clip(1, 238).astype(int)
salary = np.exp(4.5 + 0.08 * years + 0.005 * hits + np.random.normal(0, 0.5, n)).clip(60, 2500)

r1 = salary[years < 4.5]
r2 = salary[(years >= 4.5) & (hits < 117.5)]
r3 = salary[(years >= 4.5) & (hits >= 117.5)]

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.set_xlim(0, 10.8)
ax.set_ylim(0, 6.5)
ax.axis("off")

q_kw = dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#d4a017", linewidth=1.5)
leaf_kw = dict(boxstyle="round,pad=0.35", facecolor="#e8f4fd", edgecolor="#4a90d9", linewidth=1.5)
edge_kw = dict(color="#333", linewidth=1.5)
yn_kw = dict(fontsize=8.5, ha="center", va="center", color="#888")

ax.text(5.2, 5.8, "Years < 4.5?", fontsize=11, ha="center", va="center", fontweight="bold", bbox=q_kw)
ax.text(2.0, 3.5, f"${np.mean(r1):,.0f}", fontsize=10, ha="center", va="center",
        fontweight="bold", color="#c0392b", bbox=leaf_kw)
ax.text(8.35, 3.85, "Hits < 117.5?", fontsize=10, ha="center", va="center", fontweight="bold", bbox=q_kw)
ax.text(6.4, 1.45, f"${np.mean(r2):,.0f}", fontsize=10, ha="center", va="center",
        fontweight="bold", color="#c0392b", bbox=leaf_kw)
ax.text(10.2, 1.45, f"${np.mean(r3):,.0f}", fontsize=10, ha="center", va="center",
        fontweight="bold", color="#c0392b", bbox=leaf_kw)

ax.plot([5.2, 2.0], [5.4, 3.9], **edge_kw)
ax.plot([5.2, 8.35], [5.4, 4.2], **edge_kw)
ax.plot([8.35, 6.4], [3.45, 1.9], **edge_kw)
ax.plot([8.35, 10.2], [3.45, 1.9], **edge_kw)

ax.text(3.35, 4.8, "có", **yn_kw)
ax.text(7.15, 5.0, "không", **yn_kw)
ax.text(7.0, 2.85, "có", **yn_kw)
ax.text(9.45, 2.85, "không", **yn_kw)

ax.annotate("nhánh", xy=(6.6, 4.9), fontsize=8, color="#c0392b", fontweight="bold",
            xytext=(7.55, 5.6), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
ax.annotate("nút trong", xy=(8.95, 4.0), fontsize=8, color="#c0392b", fontweight="bold",
            xytext=(10.35, 4.95), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
ax.annotate("lá", xy=(1.45, 3.72), fontsize=8, color="#c0392b", fontweight="bold",
            xytext=(0.55, 4.85), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

fig.tight_layout(pad=0.3)
fig.savefig("../hitters-tree.svg", format="svg", bbox_inches="tight")
print("Saved hitters-tree.svg")
