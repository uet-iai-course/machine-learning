"""Wisdom of Crowds: Oscar voting simulation.

50 voters, 10 categories, 4 nominations each.
15 informed voters (correct with probability p), 35 uninformed (random = 0.25).
Compare: individual voter score vs consensus (majority vote) score.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n_voters = 50
n_informed = 15
n_categories = 10
n_choices = 4
n_trials = 500

p_values = np.linspace(0.25, 1.0, 16)

individual_means, individual_stds = [], []
consensus_means, consensus_stds = [], []

for p in p_values:
    ind_scores = []
    con_scores = []
    for _ in range(n_trials):
        # correct answer is always 0
        votes = np.zeros((n_voters, n_categories), dtype=int)
        for v in range(n_voters):
            prob_correct = p if v < n_informed else 1.0 / n_choices
            for c in range(n_categories):
                if rng.random() < prob_correct:
                    votes[v, c] = 0  # correct
                else:
                    votes[v, c] = rng.integers(1, n_choices)  # wrong

        # Individual: pick random voter
        idx = rng.integers(n_voters)
        ind_scores.append(np.sum(votes[idx] == 0))

        # Consensus: majority vote per category
        correct = 0
        for c in range(n_categories):
            counts = np.bincount(votes[:, c], minlength=n_choices)
            winner = np.argmax(counts)
            if winner == 0:
                correct += 1
        con_scores.append(correct)

    individual_means.append(np.mean(ind_scores))
    individual_stds.append(np.std(ind_scores))
    consensus_means.append(np.mean(con_scores))
    consensus_stds.append(np.std(con_scores))

individual_means = np.array(individual_means)
individual_stds = np.array(individual_stds)
consensus_means = np.array(consensus_means)
consensus_stds = np.array(consensus_stds)

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.errorbar(p_values, consensus_means, yerr=consensus_stds,
            fmt="o-", color="#e8732a", markeredgecolor="#c45a1a",
            markersize=6, capsize=3, linewidth=1.5, label="Consensus")
ax.errorbar(p_values, individual_means, yerr=individual_stds,
            fmt="o--", color="#5aaa44", markeredgecolor="#3d7a2e",
            markersize=6, capsize=3, linewidth=1.5, label="Individual")

ax.set_xlabel("P — Xác suất người hiểu biết chọn đúng", fontsize=10)
ax.set_ylabel("Số câu đúng kỳ vọng (trên 10)", fontsize=10)
ax.set_xlim(0.2, 1.05)
ax.set_ylim(0, 10.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("../wisdom-crowds.svg", format="svg", bbox_inches="tight")
print("Saved wisdom-crowds.svg")
