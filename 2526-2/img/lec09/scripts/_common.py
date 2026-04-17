"""Shared style helpers for lecture 09 figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colors consistent with lecture 08 / course theme
C_POS = "#2c6ea3"      # class +1 (blue)
C_NEG = "#c0392b"      # class -1 (pink/red)
C_HYP = "#333333"      # hyperplane line
C_MARGIN = "#999999"   # margin band
C_ARROW = "#e8732a"    # annotation arrows / orange accent

# Default rc
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_separable_2d(n_per=20, sep=2.5, noise=0.6, seed=42):
    """Two-class 2D dataset that is linearly separable with margin."""
    rng = np.random.RandomState(seed)
    mean_pos = np.array([ sep / 2,  sep / 2])
    mean_neg = np.array([-sep / 2, -sep / 2])
    X_pos = rng.randn(n_per, 2) * noise + mean_pos
    X_neg = rng.randn(n_per, 2) * noise + mean_neg
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_per), -np.ones(n_per)])
    return X, y


def scatter_two_class(ax, X, y, s=40, alpha=0.85, legend=False):
    m_pos = y > 0
    ax.scatter(X[m_pos, 0], X[m_pos, 1], c=C_POS, s=s, alpha=alpha,
               edgecolors="white", linewidths=0.5, label="class +1" if legend else None)
    ax.scatter(X[~m_pos, 0], X[~m_pos, 1], c=C_NEG, s=s, alpha=alpha,
               edgecolors="white", linewidths=0.5, label="class \u22121" if legend else None)
