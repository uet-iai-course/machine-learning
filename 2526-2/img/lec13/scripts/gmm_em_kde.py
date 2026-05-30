"""Sinh 3 hình cho section "Mô hình sinh cổ điển" (Bài 13):

- gmm-clusters.svg   : GMM khớp dữ liệu 2D — mỗi cụm là một Gauss (ellipse).
- em-iterations.svg  : EM hội tụ qua 3 mốc (khởi tạo → vài vòng → hội tụ).
- kde-bandwidth.svg  : KDE 1D với 3 bandwidth (nhỏ/vừa/lớn).

Dùng .conda/bin/python img/lec13/scripts/gmm_em_kde.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PALETTE, apply_style, save_svg  # noqa: E402

OUT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(7)

# 3 cụm màu — dùng palette deck
COLORS = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]


def make_data():
    """3 cụm Gauss 2D — trả về (X, true_means, true_covs)."""
    means = [np.array([0.0, 0.0]), np.array([3.4, 2.6]), np.array([3.2, -2.2])]
    covs = [
        np.array([[1.0, 0.4], [0.4, 0.7]]),
        np.array([[0.7, -0.3], [-0.3, 0.9]]),
        np.array([[1.1, 0.2], [0.2, 0.5]]),
    ]
    sizes = [120, 100, 90]
    pts = [RNG.multivariate_normal(m, c, n) for m, c, n in zip(means, covs, sizes)]
    X = np.vstack(pts)
    labels = np.concatenate([[i] * n for i, n in enumerate(sizes)])
    return X, labels, means, covs


def draw_ellipse(ax, mean, cov, color, lw=2.2, alpha=1.0, ls="-", zorder=3):
    """Vẽ ellipse 2-sigma cho một Gauss."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for k in (2.0,):  # 2-sigma
        w, h = 2 * k * np.sqrt(vals)
        e = Ellipse(mean, w, h, angle=angle, fill=False,
                    edgecolor=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder)
        ax.add_patch(e)


# ---------- 1. GMM clusters ----------

def make_gmm():
    apply_style()
    X, labels, means, covs = make_data()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i in range(3):
        m = labels == i
        ax.scatter(X[m, 0], X[m, 1], s=20, color=COLORS[i],
                   alpha=0.45, edgecolors="none", zorder=2)
        draw_ellipse(ax, means[i], covs[i], COLORS[i], lw=2.6, zorder=3)
        ax.plot(*means[i], marker="x", color=COLORS[i], ms=11, mew=3, zorder=4)
    ax.set_title(r"$p(x)=\sum_k \pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k)$",
                 fontsize=15, pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Mỗi cụm = một thành phần Gauss; trộn lại thành p(x)",
                  fontsize=12, color=PALETTE["muted"])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save_svg(fig, str(OUT / "gmm-clusters.svg"))


# ---------- 2. EM iterations ----------

def _gaussian_pdf(X, mean, cov):
    d = X - mean
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    e = np.einsum("ni,ij,nj->n", d, inv, d)
    return np.exp(-0.5 * e) / (2 * np.pi * np.sqrt(det))


def run_em(X, K=3, n_iter=30, seed=3):
    """EM tối thiểu cho GMM — trả về snapshot (means, covs) tại các mốc."""
    rng = np.random.default_rng(seed)
    n = len(X)
    # init: chọn ngẫu nhiên + lệch để thấy quá trình hội tụ
    means = X[rng.choice(n, K, replace=False)] + rng.normal(0, 0.6, (K, 2))
    covs = [np.eye(2) * 2.0 for _ in range(K)]
    pis = np.ones(K) / K
    snaps = {}
    for it in range(n_iter + 1):
        if it in (0, 4, n_iter):
            snaps[it] = ([m.copy() for m in means], [c.copy() for c in covs])
        # E-step
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = pis[k] * _gaussian_pdf(X, means[k], covs[k])
        resp /= resp.sum(1, keepdims=True) + 1e-12
        # M-step
        Nk = resp.sum(0)
        for k in range(K):
            means[k] = (resp[:, k, None] * X).sum(0) / Nk[k]
            d = X - means[k]
            covs[k] = (resp[:, k, None, None] * np.einsum("ni,nj->nij", d, d)).sum(0) / Nk[k]
            covs[k] += np.eye(2) * 1e-3
        pis = Nk / n
    return snaps


def make_em():
    apply_style()
    X, _, _, _ = make_data()
    snaps = run_em(X)
    mocs = [(0, "Khởi tạo"), (4, "Sau 4 vòng"), (30, "Hội tụ")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    for ax, (it, title) in zip(axes, mocs):
        means, covs = snaps[it]
        ax.scatter(X[:, 0], X[:, 1], s=12, color=PALETTE["muted"],
                   alpha=0.32, edgecolors="none", zorder=2)
        for k in range(3):
            draw_ellipse(ax, means[k], covs[k], COLORS[k], lw=2.4, zorder=3)
            ax.plot(*means[k], marker="x", color=COLORS[k], ms=9, mew=2.6, zorder=4)
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("EM lặp: E-step (gán trách nhiệm) → M-step (cập nhật Gauss)",
                 fontsize=14, y=1.04)
    save_svg(fig, str(OUT / "em-iterations.svg"))


# ---------- 3. KDE bandwidth ----------

def make_kde():
    apply_style()
    # dữ liệu 1D đa đỉnh
    data = np.concatenate([
        RNG.normal(-2.2, 0.5, 60),
        RNG.normal(0.6, 0.4, 40),
        RNG.normal(3.0, 0.7, 50),
    ])
    xs = np.linspace(-5, 6, 400)

    def kde(h):
        u = (xs[:, None] - data[None, :]) / h
        k = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        return k.sum(1) / (len(data) * h)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    cfg = [(0.12, "h nhỏ — quá gồ ghề", PALETTE["red"]),
           (0.45, "h vừa — hợp lý", PALETTE["green"]),
           (1.4, "h lớn — quá mượt", PALETTE["orange"])]
    for ax, (h, title, col) in zip(axes, cfg):
        ax.fill_between(xs, kde(h), color=col, alpha=0.18, zorder=1)
        ax.plot(xs, kde(h), color=col, lw=2.4, zorder=3)
        ax.plot(data, np.full_like(data, -0.01), "|", color=PALETTE["ink"],
                ms=8, alpha=0.5, zorder=2)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylim(-0.03, None)
        ax.spines["left"].set_visible(False)
    fig.suptitle(r"KDE: $\hat p(x)=\frac{1}{nh}\sum_i K\!\left(\frac{x-x_i}{h}\right)$"
                 " — bandwidth h kiểm soát độ mượt", fontsize=13, y=1.05)
    save_svg(fig, str(OUT / "kde-bandwidth.svg"))


def main():
    make_gmm()
    make_em()
    make_kde()


if __name__ == "__main__":
    main()
