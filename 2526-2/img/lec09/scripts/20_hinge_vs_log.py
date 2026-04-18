"""Slide 32 — Hinge loss vs log loss over y*f(x)."""
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-3, 3, 300)
hinge = np.maximum(0, 1 - z)
log_loss = np.log(1 + np.exp(-z))
zero_one = (z < 0).astype(float)

fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(z, zero_one, color="#888", linestyle=":", linewidth=1.8, label="0-1 loss")
ax.plot(z, hinge, color="#c0392b", linewidth=2.2, label="Hinge (SVM)")
ax.plot(z, log_loss, color="#2c6ea3", linewidth=2.2, label="Log (hồi quy logistic)")

ax.axvline(1, color="#d4a017", linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(1.08, 2.6, "điểm ngừng phạt\ncủa hinge", fontsize=9, color="#d4a017")

ax.set_xlim(-3, 3)
ax.set_ylim(-0.1, 3.2)
ax.set_xlabel(r"$y \cdot f(x)$")
ax.set_ylabel("Loss")
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("../hinge-vs-log.svg", format="svg", bbox_inches="tight")
print("Saved hinge-vs-log.svg")
