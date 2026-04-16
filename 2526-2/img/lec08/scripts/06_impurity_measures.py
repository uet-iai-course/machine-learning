"""Plot impurity measures: Gini, entropy, misclassification error."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 500)

gini = 2 * p * (1 - p)
entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2  # scaled to max 0.5
misclass = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(p, entropy, color="#e74c3c", linewidth=2, label="Entropy (chuẩn hoá)")
ax.plot(p, gini, color="#2ecc71", linewidth=2, label="Gini index")
ax.plot(p, misclass, color="#3498db", linewidth=2, label="Misclassification error")

ax.set_xlabel(r"$\hat{p}_{m1}$", fontsize=11)
ax.set_ylabel("Giá trị", fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.55)
ax.legend(fontsize=8.5, loc="upper center")
ax.grid(True, alpha=0.3)

fig.tight_layout(pad=0.5)
fig.savefig("../impurity-measures.svg", format="svg", bbox_inches="tight")
print("Saved impurity-measures.svg")
