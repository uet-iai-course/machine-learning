"""Simple 20-questions tree SVG using matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Style
box_kw = dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#333", linewidth=1.5)
leaf_kw = dict(boxstyle="round,pad=0.4", facecolor="#e8f4fd", edgecolor="#4a90d9", linewidth=1.5)
font_q = dict(fontsize=11, ha="center", va="center", fontweight="bold", bbox=box_kw)
font_a = dict(fontsize=11, ha="center", va="center", fontweight="bold", bbox=leaf_kw, color="#4a90d9")
font_yn = dict(fontsize=9, ha="center", va="center", color="#888")

# Nodes
nodes = {
    "root": (5, 5.2, "Có lông?"),
    "q1":   (2.5, 3.2, "Có sủa?"),
    "q2":   (7.5, 3.2, "Biết bay?"),
}
leaves = {
    "dog":  (1, 1.2, "Chó"),
    "cat":  (4, 1.2, "Mèo"),
    "bird": (6, 1.2, "Chim"),
    "fish": (9, 1.2, "Cá"),
}

for _, (x, y, txt) in nodes.items():
    ax.text(x, y, txt, **font_q)
for _, (x, y, txt) in leaves.items():
    ax.text(x, y, txt, **font_a)

# Edges
edge_kw = dict(color="#333", linewidth=1.5)
def draw_edge(x1, y1, x2, y2, label, label_side="left"):
    ax.plot([x1, x2], [y1 - 0.35, y2 + 0.35], **edge_kw)
    mx = (x1 + x2) / 2
    my = (y1 - 0.35 + y2 + 0.35) / 2
    offset = -0.35 if label_side == "left" else 0.35
    ax.text(mx + offset, my, label, **font_yn)

draw_edge(5, 5.2, 2.5, 3.2, "có", "left")
draw_edge(5, 5.2, 7.5, 3.2, "không", "right")
draw_edge(2.5, 3.2, 1, 1.2, "có", "left")
draw_edge(2.5, 3.2, 4, 1.2, "không", "right")
draw_edge(7.5, 3.2, 6, 1.2, "có", "left")
draw_edge(7.5, 3.2, 9, 1.2, "không", "right")

fig.tight_layout(pad=0.3)
fig.savefig("../20questions-tree.svg", format="svg", bbox_inches="tight")
print("Saved 20questions-tree.svg")
