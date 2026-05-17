"""Sigmoid vs ReLU — 2 panels side by side với công thức ở góc."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def main():
    z = np.linspace(-6, 6, 400)
    sig = 1 / (1 + np.exp(-z))
    relu = np.maximum(0, z)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # Sigmoid panel
    ax = axes[0]
    ax.plot(z, sig, color=C_BLUE, lw=2.5)
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.axvline(0, color=C_GRAY, lw=0.5)
    ax.set_title("Sigmoid")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\sigma(z)$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2, ls="--")
    ax.text(0.04, 0.94, r"$\sigma(z) = \dfrac{1}{1 + e^{-z}}$",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor="white", edgecolor=C_BLUE, lw=1.2))

    # ReLU panel
    ax = axes[1]
    ax.plot(z, relu, color=C_RED, lw=2.5)
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.axvline(0, color=C_GRAY, lw=0.5)
    ax.set_title("ReLU")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$R(z)$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, 6.5)
    ax.grid(True, alpha=0.2, ls="--")
    ax.text(0.04, 0.94, r"$R(z) = \max(0,\, z)$",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor="white", edgecolor=C_RED, lw=1.2))

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "sigmoid-vs-relu.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
