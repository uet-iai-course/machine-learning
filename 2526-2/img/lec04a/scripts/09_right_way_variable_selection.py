"""
Recreate the "Right way" histogram for variable selection without leakage.

Procedure:
1. Simulate n=50 samples with p=5000 Gaussian predictors independent of the label.
2. Randomly hold out 10 samples.
3. Select the 100 predictors with the largest sample correlation with the label
   using only the remaining training samples (the right way).
4. Recompute correlations for those selected predictors on the held-out subset.
5. Plot the histogram of held-out correlations.

Output: ../cv-right-way-variable-selection.svg
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


SEED = 494
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

    held_out = rng.choice(N, size=SUBSET_SIZE, replace=False)
    train = np.setdiff1d(np.arange(N), held_out)

    train_corr = corr_cols(x[train], y[train])
    selected = np.argsort(train_corr)[-NUM_SELECTED:]
    held_out_corr = corr_cols(x[held_out][:, selected], y[held_out])

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bins = np.linspace(-1.0, 1.0, 11)
    ax.hist(held_out_corr, bins=bins, color="#007300", edgecolor="black", linewidth=1.2)

    ax.set_title("Right way", fontsize=20, fontweight="bold", pad=10)
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
    output = Path(__file__).resolve().parent.parent / "cv-right-way-variable-selection.svg"
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Mean held-out correlation: {held_out_corr.mean():.4f}")


if __name__ == "__main__":
    main()
