"""Variable importance bar chart for Heart dataset using Random Forest."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

# Load Heart
heart = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto")
X = heart.data.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c)
y = (heart.target == "present").astype(int) if heart.target.dtype == object or hasattr(heart.target, "cat") else (heart.target.astype(int) > 0).astype(int)
feature_names = X.columns.tolist()

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
# Scale to max = 100
importances = importances / importances.max() * 100

# Sort ascending for horizontal bar chart
order = np.argsort(importances)
sorted_names = [feature_names[i] for i in order]
sorted_imp = importances[order]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.barh(range(len(sorted_names)), sorted_imp, color="#e74c3c", height=0.7)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=8)
ax.set_xlabel("Variable Importance", fontsize=10)
ax.set_xlim(0, 110)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout(pad=0.5)
fig.savefig("../variable-importance.svg", format="svg", bbox_inches="tight")
print("Saved variable-importance.svg")
