"""
Generate bootstrap-resampling-diagram.svg — pure SVG diagram showing
bootstrap resampling from a 3-row dataset (ISLR Fig 5.11 style).

Output: ../bootstrap-resampling-diagram.svg
"""
import math

W, H = 530, 360

# ── Table geometry ─────────────────────────────────────────────────────────────
COL_W = [28, 40, 40]   # Obs, X, Y column widths
HDR_H = 20
ROW_H = 18
TW    = sum(COL_W)               # 108
TH    = HDR_H + 3 * ROW_H        # 74

# ── Colours ────────────────────────────────────────────────────────────────────
HDR  = "#c8d8ea"
EVEN = "#f0f4f8"
ODD  = "#ffffff"
STR  = "#999999"
TC   = "#222222"
AC   = "#555555"

# ── Positions ──────────────────────────────────────────────────────────────────
OX, OY = 10, (H - TH) // 2      # original table top-left; centre vertically
BX     = 215                     # bootstrap tables x
BY1    = 18
BY2    = (H - TH) // 2
BYB    = H - 18 - TH

OCY    = OY  + TH // 2           # vertical centres
BC1    = BY1 + TH // 2
BC2    = BY2 + TH // 2
BCB    = BYB + TH // 2
OR     = OX  + TW                # right edge of original table

AX     = BX + TW + 18            # x of α̂ labels

# ── Dataset ────────────────────────────────────────────────────────────────────
ORIG = [(1, 4.3, 2.4), (2, 2.1, 1.1), (3, 5.3, 2.8)]
Z1   = [(3, 5.3, 2.8), (3, 5.3, 2.8), (3, 5.3, 2.8)]
Z2   = [(2, 2.1, 1.1), (3, 5.3, 2.8), (1, 4.3, 2.4)]
ZB   = [(2, 2.1, 1.1), (1, 4.3, 2.4), (1, 4.3, 2.4)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def cell(x, y, w, h, text, fill, bold=False, fs=10):
    fw = "bold" if bold else "normal"
    tx, ty = x + w / 2, y + h * 0.68
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{STR}" stroke-width="0.7"/>'
        f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
        f'font-size="{fs}" font-weight="{fw}" fill="{TC}">{text}</text>\n'
    )

def draw_table(x, y, rows):
    s = ""
    xs = [x + sum(COL_W[:i]) for i in range(3)]
    for i, (hdr, w) in enumerate(zip(["Obs", "X", "Y"], COL_W)):
        s += cell(xs[i], y, w, HDR_H, hdr, HDR, bold=True)
    for r, row in enumerate(rows):
        fill = ODD if r % 2 == 0 else EVEN
        ry = y + HDR_H + r * ROW_H
        for i, (val, w) in enumerate(zip(row, COL_W)):
            s += cell(xs[i], ry, w, ROW_H, val, fill)
    return s

def arrow(x1, y1, x2, y2, shorten=5):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    xe, ye = x2 - shorten * ux, y2 - shorten * uy
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{xe:.1f}" y2="{ye:.1f}" '
        f'stroke="{AC}" stroke-width="1.4" marker-end="url(#arr)"/>\n'
    )

def zstar_label(x, y, sup):
    """Z with superscript *N rendered in SVG."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-size="10" font-style="italic" fill="#333">'
        f'Z<tspan dy="-4" font-size="8" font-style="normal">*{sup}</tspan></text>\n'
    )

def ahat_label(x, y, sup):
    """α̂ with superscript *N. α = U+03B1, combining circumflex = U+0302."""
    return (
        f'<text x="{x}" y="{y}" font-size="11" font-style="italic" fill="#333">'
        f'\u03b1\u0302'
        f'<tspan dy="-4" font-size="8" font-style="normal">*{sup}</tspan></text>\n'
    )

def vdots(x, y):
    """Vertical ellipsis centred at (x,y)."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-size="18" fill="#777">\u22ee</text>\n'
    )

# ── Arrow midpoints for labels ─────────────────────────────────────────────────
mx = (OR + BX) / 2
my1 = (OCY + BC1) / 2
my2 = (OCY + BC2) / 2
myB = (OCY + BCB) / 2

# ── Build SVG ─────────────────────────────────────────────────────────────────
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="sans-serif">\n',
    # arrowhead marker
    '<defs><marker id="arr" markerWidth="8" markerHeight="6" '
    'refX="7" refY="3" orient="auto">'
    '<path d="M0,0 L0,6 L8,3 z" fill="#555"/></marker></defs>\n',

    # ── Original table ────────────────────────────────────────────────────────
    draw_table(OX, OY, ORIG),
    # "Original Data (Z)" caption below
    f'<text x="{OX + TW/2:.0f}" y="{OY + TH + 15}" text-anchor="middle" '
    f'font-size="9" fill="#444">Original Data</text>\n',
    f'<text x="{OX + TW/2:.0f}" y="{OY + TH + 26}" text-anchor="middle" '
    f'font-size="9" fill="#444">(Z)</text>\n',

    # ── Arrows original → bootstrap ───────────────────────────────────────────
    arrow(OR, OCY, BX, BC1),
    arrow(OR, OCY, BX, BC2),
    arrow(OR, OCY, BX, BCB),

    # ── Arrow labels ──────────────────────────────────────────────────────────
    zstar_label(mx, my1 - 7, "1"),
    zstar_label(mx, my2 - 7, "2"),
    zstar_label(mx, myB + 18, "B"),

    # ── Bootstrap tables ──────────────────────────────────────────────────────
    draw_table(BX, BY1, Z1),
    draw_table(BX, BY2, Z2),
    draw_table(BX, BYB, ZB),

    # ── Vertical dots between Z*2 and Z*B ────────────────────────────────────
    vdots(BX + TW / 2, (BY2 + TH + BYB) / 2 + 5),

    # ── Small arrows → α̂ ─────────────────────────────────────────────────────
    arrow(BX + TW + 2, BC1, AX - 2, BC1, shorten=3),
    arrow(BX + TW + 2, BC2, AX - 2, BC2, shorten=3),
    arrow(BX + TW + 2, BCB, AX - 2, BCB, shorten=3),

    # ── α̂ labels ──────────────────────────────────────────────────────────────
    ahat_label(AX, BC1 + 4, "1"),
    ahat_label(AX, BC2 + 4, "2"),
    ahat_label(AX, BCB + 4, "B"),

    # ── Vertical dots between α̂*2 and α̂*B ───────────────────────────────────
    vdots(AX + 14, (BC2 + BCB) / 2 + 5),

    '</svg>',
]

svg = "".join(parts)
out = "../bootstrap-resampling-diagram.svg"
with open(out, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved: {out}")
