"""
Recreate the "Wrong way" histogram for variable selection leakage.

Procedure:
1. Simulate n=50 samples with p=5000 Gaussian predictors independent of the label.
2. Select the 100 predictors with the largest sample correlation with the label
   using the full dataset (the wrong way).
3. Draw a random subset of 10 samples and recompute correlations for the selected
   predictors on that subset.
4. Plot the histogram of those correlations.

Output: ../cv-wrong-way-variable-selection.svg
"""

import os
from pathlib import Path

TMP_CACHE = Path("/tmp/codex-mpl-cache")
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 409
N = 50
P = 5000
NUM_SELECTED = 100
SUBSET_SIZE = 10


def corr_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return column-wise Pearson correlations between x and y."""
    y_centered = y - y.mean()
    x_centered = x - x.mean(axis=0)
    numerator = (x_centered * y_centered[:, None]).sum(axis=0)
    denominator = np.sqrt((x_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
    return numerator / denominator


def main() -> None:
    rng = np.random.default_rng(SEED)

    y = np.r_[np.zeros(N // 2), np.ones(N - N // 2)]
    rng.shuffle(y)
    x = rng.normal(size=(N, P))

    full_corr = corr_cols(x, y)
    selected = np.argsort(full_corr)[-NUM_SELECTED:]

    subset = rng.choice(N, size=SUBSET_SIZE, replace=False)
    subset_corr = corr_cols(x[subset][:, selected], y[subset])

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bins = np.linspace(-1.0, 1.0, 11)
    ax.hist(subset_corr, bins=bins, color="#ff120a", edgecolor="black", linewidth=1.2)

    ax.set_title("Wrong way", fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Correlations of Selected Predictors with Outcome", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "cv-wrong-way-variable-selection.svg"
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Mean correlation on subset: {subset_corr.mean():.4f}")


if __name__ == "__main__":
    main()
