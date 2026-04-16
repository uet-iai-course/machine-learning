"""Euclidean vs correlation-based distance illustration.

One plot with 3 profiles + a table of pairwise distances.
- Obs A & B: same shape (sin), different magnitude → corr ≈ 1, Euclid lớn
- Obs A & C: different shape (flat+noise), similar magnitude → corr thấp, Euclid nhỏ
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
p = 8  # number of features (like product categories)
x = np.arange(p)
labels = [f"$x_{{{i+1}}}$" for i in range(p)]

# Obs A: rising-falling pattern, high values
A = np.array([3, 7, 12, 15, 14, 10, 6, 2], dtype=float)
# Obs B: SAME shape, scaled down (corr ≈ 1, Euclid far)
B = A * 0.3 + 0.5
# Obs C: DIFFERENT shape, similar magnitude to A (corr low, Euclid closer)
C = np.array([10, 8, 11, 5, 12, 9, 13, 7], dtype=float)

# Compute distances
euclid_AB = np.linalg.norm(A - B)
euclid_AC = np.linalg.norm(A - C)
corr_AB = 1 - np.corrcoef(A, B)[0, 1]
corr_AC = 1 - np.corrcoef(A, C)[0, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                gridspec_kw={"width_ratios": [2, 1]})

# ── Left: profiles ──
ax1.plot(x, A, "o-", color="#4a90d9", linewidth=2.2, markersize=7, label="Obs A")
ax1.plot(x, B, "s--", color="#e8732a", linewidth=2.2, markersize=7, label="Obs B")
ax1.plot(x, C, "^:", color="#5aaa44", linewidth=2.2, markersize=7, label="Obs C")

# Fill between A and B to show Euclidean gap
ax1.fill_between(x, A, B, alpha=0.08, color="#e8732a")
# Annotation: A & B same shape
ax1.annotate("cùng hình dạng,\nkhác độ lớn",
             xy=(3, (A[3] + B[3]) / 2), fontsize=8, color="#e8732a",
             ha="center", fontstyle="italic")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Giá trị", fontsize=10)
ax1.set_xlabel("Đặc trưng", fontsize=10)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_title("3 quan sát (profiles)", fontsize=11, fontweight="bold")

# ── Right: distance table ──
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

table_data = [
    ["", "Euclid", "Correlation"],
    ["A ↔ B", f"{euclid_AB:.1f}", f"{corr_AB:.3f}"],
    ["A ↔ C", f"{euclid_AC:.1f}", f"{corr_AC:.3f}"],
]

# Colors for cells to highlight the key insight
# Euclid: A↔B large (red), A↔C small (green)
# Corr:   A↔B small (green), A↔C large (red)
cell_colors = [
    ["#f5f5f5", "#f5f5f5", "#f5f5f5"],
    ["#f5f5f5", "#f8d7da", "#d4edda"],  # Euclid AB=big(red), Corr AB=small(green)
    ["#f5f5f5", "#d4edda", "#f8d7da"],  # Euclid AC=small(green), Corr AC=big(red)
]

table = ax2.table(cellText=table_data, cellColours=cell_colors,
                  loc="center", cellLoc="center",
                  bbox=[0.05, 0.2, 0.9, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(11)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#ccc")
    if row == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e8e8e8")

ax2.set_title("Khoảng cách giữa các cặp", fontsize=11, fontweight="bold", pad=10)

# Key insight text below table
ax2.text(0.5, 0.08,
         "Euclid: gần nếu giá trị gần\nCorrelation: gần nếu hình dạng giống",
         fontsize=9, ha="center", va="center", color="#333",
         fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6", edgecolor="#ccc"))

fig.tight_layout(pad=1.0)
fig.savefig("../dissimilarity-comparison.svg", format="svg", bbox_inches="tight")
print(f"Euclid: A↔B={euclid_AB:.1f}, A↔C={euclid_AC:.1f}")
print(f"Corr:   A↔B={corr_AB:.4f}, A↔C={corr_AC:.4f}")
print("Saved dissimilarity-comparison.svg")
