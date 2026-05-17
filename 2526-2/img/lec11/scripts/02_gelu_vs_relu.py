"""GELU vs ReLU — overlay 2 đường, highlight vùng GELU cho phép giá trị âm."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from _common import C_BLUE, C_RED, C_GRAY, save_svg


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


def main():
    z = np.linspace(-4, 4, 400)
    relu_vals = np.maximum(0, z)
    gelu_vals = gelu(z)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Highlight negative dip region of GELU
    ax.fill_between(z, gelu_vals, 0, where=gelu_vals < 0,
                    color="#fde2c4", alpha=0.7,
                    label="GELU cho phép giá trị âm")

    ax.plot(z, relu_vals, color=C_BLUE, lw=2.4, label="ReLU")
    ax.plot(z, gelu_vals, color=C_RED, lw=2.4, label="GELU")

    ax.axhline(0, color=C_GRAY, lw=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.6)

    ax.set_title("GELU vs ReLU")
    ax.set_xlabel("z")
    ax.set_ylabel("Activation output")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.2, ls="--")
    ax.legend(loc="upper left", frameon=False)

    # Annotation pointing to dip
    ax.annotate("Không có dead neuron\nở vùng $x \\approx -0.5$",
                xy=(-0.7, -0.17), xytext=(-3.5, 2.4),
                fontsize=10, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "gelu-vs-relu.svg"
    save_svg(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
