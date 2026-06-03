"""Sinh 3 sơ đồ kiến trúc (hand-crafted SVG) cho Bài 13:

- vae-architecture.svg        : Encoder → (μ,σ) → z → Decoder
- diffusion-process.svg       : chuỗi ảnh → nhiễu (xuôi) / nhiễu → ảnh (ngược)
- stable-diffusion-pipeline.svg : prompt → CLIP → U-Net (ẩn) → VAE decoder → ảnh

Hand SVG (không matplotlib) cho sơ đồ — sạch, nhẹ, đúng palette deck.
Dùng .conda/bin/python img/lec13/scripts/arch_diagrams.py
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent

# Palette đồng bộ lecture-style.css
BLUE = "#1E93AB"
GREEN = "#4a9d3f"
ORANGE = "#e8732a"
PURPLE = "#8e4ec6"
INK = "#1a2332"
MUTED = "#6b7280"
GRID = "#cfd6dc"
BG_BLUE = "#f4fafc"
BG_GREEN = "#f1f8ef"
BG_ORANGE = "#fef5ee"
BG_PURPLE = "#f7f0fb"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif")


def marker_def(mid, color):
    """Đầu mũi tên — neo ở ĐÁY (refX=0): đáy tam giác đặt đúng điểm cuối
    line, đỉnh nhô về phía trước (theo orient=auto).

    Kết hợp với việc rút ngắn line đi ~chiều dài mũi tên (xem ARROW_LEN /
    arrow()), thân line dừng đúng tại đáy — phần RỘNG nhất của mũi tên —
    nên không bao giờ lộ "mấu" ở đỉnh. viewBox cho mapping nhất quán.
    """
    return (f'<marker id="{mid}" viewBox="0 0 10 10" markerWidth="6" '
            f'markerHeight="6" refX="0" refY="5" orient="auto" '
            f'markerUnits="strokeWidth">'
            f'<path d="M0,1 L10,5 L0,9 Z" fill="{color}"/></marker>')


def header(w, h):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">\n'
            f'<defs>{marker_def("arr", INK)}</defs>\n')


def box(x, y, w, h, fill, stroke, label, sub="", fs=17, rx=8):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>\n')
    if sub:
        s += (f'<text x="{x+w/2}" y="{y+h/2-4}" text-anchor="middle" '
              f'font-size="{fs}" font-weight="600" fill="{INK}">{label}</text>\n')
        s += (f'<text x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" '
              f'font-size="12" fill="{MUTED}">{sub}</text>\n')
    else:
        s += (f'<text x="{x+w/2}" y="{y+h/2+6}" text-anchor="middle" '
              f'font-size="{fs}" font-weight="600" fill="{INK}">{label}</text>\n')
    return s


def trapezoid(x, y, w, h, narrow_right, fill, stroke, label):
    """Hình thang: narrow_right=True → encoder (hẹp dần phải)."""
    inset = h * 0.28
    if narrow_right:
        pts = f"{x},{y} {x+w},{y+inset} {x+w},{y+h-inset} {x},{y+h}"
    else:
        pts = f"{x},{y+inset} {x+w},{y} {x+w},{y+h} {x},{y+h-inset}"
    s = (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
         f'stroke-width="2.2"/>\n')
    s += (f'<text x="{x+w/2}" y="{y+h/2+5}" text-anchor="middle" '
          f'font-size="15" font-weight="600" fill="{INK}">{label}</text>\n')
    return s


# Rút ngắn line đi đoạn này (≈ chiều dài mũi tên trên canvas: markerWidth 6
# × stroke 2.2 ≈ 13) để thân line dừng ở đáy mũi tên, đỉnh nhô tới đích.
ARROW_LEN = 12.0


def _retract(x1, y1, x2, y2, r=ARROW_LEN):
    """Kéo (x2,y2) về phía (x1,y1) một đoạn r px."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    return x2 - dx / L * r, y2 - dy / L * r


def arrow(x1, y1, x2, y2, label="", color=INK, dash=False):
    ex, ey = _retract(x1, y1, x2, y2)
    d = 'stroke-dasharray="5,4" ' if dash else ""
    s = (f'<line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" '
         f'stroke-width="2.2" {d}marker-end="url(#arr)"/>\n')
    if label:
        s += (f'<text x="{(x1+x2)/2}" y="{y1-9}" text-anchor="middle" '
              f'font-size="13" fill="{color}">{label}</text>\n')
    return s


def noisy_square(x, y, sz, noise, seed):
    """Ô vuông ảnh với mức nhiễu (0=ảnh sạch, 1=nhiễu thuần)."""
    import math
    # nền chuyển từ xanh nhạt (ảnh) sang xám (nhiễu)
    t = noise
    r = int(0xf4 + (0xcf - 0xf4) * t)
    g = int(0xfa + (0xd6 - 0xfa) * t)
    b = int(0xfc + (0xdc - 0xfc) * t)
    fill = f"rgb({r},{g},{b})"
    s = (f'<rect x="{x}" y="{y}" width="{sz}" height="{sz}" rx="4" '
         f'fill="{fill}" stroke="{GRID}" stroke-width="1.5"/>\n')
    # vẽ "vật thể" (vòng tròn xanh) mờ dần khi nhiễu tăng
    obj_alpha = max(0.0, 1 - 1.15 * t)
    if obj_alpha > 0.02:
        s += (f'<circle cx="{x+sz/2}" cy="{y+sz/2}" r="{sz*0.26}" '
              f'fill="{BLUE}" opacity="{obj_alpha:.2f}"/>\n')
    # chấm nhiễu — số lượng tăng theo noise
    n = int(noise * 16)
    rng = (seed * 9301 + 49297)
    for i in range(n):
        rng = (rng * 9301 + 49297) % 233280
        px = x + 4 + (rng / 233280) * (sz - 8)
        rng = (rng * 9301 + 49297) % 233280
        py = y + 4 + (rng / 233280) * (sz - 8)
        rng = (rng * 9301 + 49297) % 233280
        shade = int(60 + (rng / 233280) * 120)
        s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.7" '
              f'fill="rgb({shade},{shade},{shade})" opacity="0.8"/>\n')
    return s


# ---------- 1. VAE ----------

def make_vae():
    W, H = 720, 250
    s = header(W, H)
    cy = 90
    bh = 80
    # x input
    s += box(20, cy, 70, bh, BG_BLUE, BLUE, "x", "ảnh vào")
    s += arrow(95, cy + bh / 2, 135, cy + bh / 2)
    # encoder
    s += trapezoid(138, cy - 8, 110, bh + 16, True, BG_BLUE, BLUE, "Encoder")
    s += arrow(253, cy + bh / 2, 290, cy + bh / 2)
    # mu, sigma
    s += box(293, cy + 2, 64, 34, "#fff", BLUE, "μ", fs=16, rx=6)
    s += box(293, cy + 44, 64, 34, "#fff", BLUE, "σ", fs=16, rx=6)
    s += arrow(360, cy + bh / 2, 398, cy + bh / 2)
    # sample z
    s += (f'<circle cx="440" cy="{cy+bh/2}" r="34" fill="{BG_GREEN}" '
          f'stroke="{GREEN}" stroke-width="2.2"/>\n')
    s += (f'<text x="440" y="{cy+bh/2-2}" text-anchor="middle" font-size="18" '
          f'font-weight="600" fill="{INK}">z</text>\n')
    s += (f'<text x="440" y="{cy+bh/2+16}" text-anchor="middle" font-size="11" '
          f'fill="{MUTED}">~ N(μ,σ²)</text>\n')
    s += arrow(476, cy + bh / 2, 514, cy + bh / 2)
    # decoder
    s += trapezoid(517, cy - 8, 110, bh + 16, False, BG_GREEN, GREEN, "Decoder")
    s += arrow(632, cy + bh / 2, 668, cy + bh / 2)
    # x_hat — vẽ "x" + nét mũ ^ riêng (ký tự tổ hợp x̂ render lệch tuỳ font)
    s += (f'<rect x="670" y="{cy}" width="44" height="{bh}" rx="8" '
          f'fill="{BG_GREEN}" stroke="{GREEN}" stroke-width="2.2"/>\n')
    s += (f'<text x="692" y="{cy+bh/2+8}" text-anchor="middle" font-size="19" '
          f'font-weight="600" fill="{INK}">x</text>\n')
    s += (f'<path d="M684,{cy+bh/2-7} L692,{cy+bh/2-13} L700,{cy+bh/2-7}" '
          f'fill="none" stroke="{INK}" stroke-width="2" stroke-linecap="round" '
          f'stroke-linejoin="round"/>\n')
    # labels
    s += (f'<text x="195" y="40" text-anchor="middle" font-size="13" '
          f'fill="{BLUE}" font-weight="600">Nén về không gian ẩn</text>\n')
    s += (f'<text x="572" y="40" text-anchor="middle" font-size="13" '
          f'fill="{GREEN}" font-weight="600">Sinh lại từ z</text>\n')
    s += (f'<text x="440" y="225" text-anchor="middle" font-size="12.5" '
          f'fill="{MUTED}">Lấy mẫu z ngẫu nhiên → decode → ảnh mới</text>\n')
    s += "</svg>\n"
    (OUT / "vae-architecture.svg").write_text(s, encoding="utf-8")
    print("Saved vae-architecture.svg")


# ---------- 2. Diffusion ----------

def make_diffusion():
    W, H = 760, 300
    s = header(W, H)
    n = 5
    sz = 92
    gap = (W - 40 - n * sz) / (n - 1)
    y = 95
    noises = [0.0, 0.28, 0.55, 0.8, 1.0]
    xs = []
    for i in range(n):
        x = 20 + i * (sz + gap)
        xs.append(x)
        s += noisy_square(x, y, sz, noises[i], seed=i + 1)
        lbl = "x₀" if i == 0 else ("xₜ" if i == n - 1 else f"x{['₁','₂','₃'][i-1]}")
        s += (f'<text x="{x+sz/2}" y="{y+sz+20}" text-anchor="middle" '
              f'font-size="14" font-weight="600" fill="{INK}">{lbl}</text>\n')
    # forward arrow (top) — rút đầu phải để thân dừng ở đáy mũi tên
    s += (f'<line x1="{xs[0]+sz}" y1="{y-22}" x2="{xs[-1]-ARROW_LEN}" y2="{y-22}" '
          f'stroke="{ORANGE}" stroke-width="2.4" marker-end="url(#arr-o)"/>\n')
    # backward arrow (bottom) — rút đầu trái
    s += (f'<line x1="{xs[-1]}" y1="{y+sz+44}" x2="{xs[0]+sz+ARROW_LEN}" y2="{y+sz+44}" '
          f'stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arr-b)"/>\n')
    s += (f'<text x="{W/2}" y="{y-32}" text-anchor="middle" font-size="14.5" '
          f'font-weight="600" fill="{ORANGE}">Quá trình xuôi: thêm nhiễu Gauss '
          f'(cố định)</text>\n')
    s += (f'<text x="{W/2}" y="{y+sz+72}" text-anchor="middle" font-size="14.5" '
          f'font-weight="600" fill="{BLUE}">Quá trình ngược: mạng học khử nhiễu '
          f'từng bước</text>\n')
    # custom colored markers
    s = s.replace(
        "</defs>",
        f'{marker_def("arr-o", ORANGE)}{marker_def("arr-b", BLUE)}</defs>')
    s += "</svg>\n"
    (OUT / "diffusion-process.svg").write_text(s, encoding="utf-8")
    print("Saved diffusion-process.svg")


# ---------- 3. Stable Diffusion pipeline ----------

def make_sd():
    W, H = 800, 230
    s = header(W, H)
    y = 80
    bh = 78
    # prompt
    s += box(16, y, 124, bh, BG_PURPLE, PURPLE, "Câu lệnh", "(prompt)", fs=15)
    s += arrow(144, y + bh / 2, 178, y + bh / 2)
    # CLIP
    s += box(178, y, 120, bh, BG_BLUE, BLUE, "CLIP", "mã hoá text", fs=15)
    s += arrow(302, y + bh / 2, 338, y + bh / 2)
    # U-Net latent
    s += box(340, y, 150, bh, BG_ORANGE, ORANGE, "U-Net khử nhiễu",
             "trong không gian ẩn ×T", fs=14)
    # loop arrow on U-Net
    s += (f'<path d="M 360 {y} q -18 -26 18 -26 q 90 0 90 13" fill="none" '
          f'stroke="{ORANGE}" stroke-width="2" marker-end="url(#arr)"/>\n')
    s += arrow(494, y + bh / 2, 530, y + bh / 2)
    # VAE decoder
    s += box(532, y, 130, bh, BG_GREEN, GREEN, "VAE decoder", "ẩn → pixel", fs=14)
    s += arrow(666, y + bh / 2, 702, y + bh / 2)
    # image out
    s += noisy_square(704, y, bh, 0.0, seed=2)
    s += (f'<text x="{704+bh/2}" y="{y+bh+18}" text-anchor="middle" '
          f'font-size="13" font-weight="600" fill="{INK}">ảnh</text>\n')
    s += (f'<text x="{W/2}" y="200" text-anchor="middle" font-size="12.5" '
          f'fill="{MUTED}">Diffuse trong không gian ẩn nhỏ của VAE → nhanh hơn '
          f'diffuse thẳng trên pixel</text>\n')
    s += "</svg>\n"
    (OUT / "stable-diffusion-pipeline.svg").write_text(s, encoding="utf-8")
    print("Saved stable-diffusion-pipeline.svg")


# ---------- 4. GAN ----------

def make_gan():
    """G (nở: nhiễu→ảnh) và D (hẹp: ảnh→1 số) đấu nhau; G chỉ học qua
    gradient lan ngược từ D (mũi tên tím đứt nét)."""
    W, H = 720, 338
    s = header(W, H)
    s = s.replace("</defs>", f'{marker_def("arr-p", PURPLE)}</defs>')

    y_top = 64      # đường sinh (trên)
    y_bot = 196     # đường dữ liệu thật (dưới)
    bh = 62

    # --- z (nhiễu) → G → G(z) ---
    s += box(22, y_top, 74, bh, BG_BLUE, BLUE, "z", "nhiễu", fs=17)
    s += arrow(96, y_top + bh / 2, 132, y_top + bh / 2)
    s += trapezoid(134, y_top - 9, 108, bh + 18, False, BG_BLUE, BLUE, "G")
    s += arrow(244, y_top + bh / 2, 282, y_top + bh / 2)
    s += noisy_square(286, y_top, bh, 0.0, seed=3)
    s += (f'<text x="{286 + bh / 2}" y="{y_top + bh + 18}" text-anchor="middle" '
          f'font-size="13" font-weight="600" fill="{BLUE}">G(z): ảnh giả</text>\n')

    # --- Tập dữ liệu → x thật ---
    s += box(22, y_bot, 96, bh, BG_GREEN, GREEN, "Dữ liệu", "thật", fs=15)
    s += arrow(118, y_bot + bh / 2, 282, y_bot + bh / 2)
    s += noisy_square(286, y_bot, bh, 0.0, seed=7)
    s += (f'<text x="{286 + bh / 2}" y="{y_bot + bh + 18}" text-anchor="middle" '
          f'font-size="13" font-weight="600" fill="{GREEN}">x: ảnh thật</text>\n')

    # --- D nhận cả hai ảnh → 1 số ---
    Dx, Dy = 430, y_top - 4
    Dw, Dh = 120, (y_bot + bh) - (y_top - 4)
    s += trapezoid(Dx, Dy, Dw, Dh, True, BG_ORANGE, ORANGE, "D")
    s += arrow(286 + bh, y_top + bh / 2, Dx + 2, y_top + 26)
    s += arrow(286 + bh, y_bot + bh / 2, Dx + 2, y_bot + bh - 26)
    Dmid = Dy + Dh / 2
    s += arrow(Dx + Dw, Dmid, Dx + Dw + 36, Dmid)
    s += box(Dx + Dw + 38, Dmid - 32, 112, 64, "#fff", ORANGE, "D(·)", "thật / giả?", fs=16)

    # --- nhãn mục ---
    s += (f'<text x="188" y="40" text-anchor="middle" font-size="13.5" '
          f'fill="{BLUE}" font-weight="600">Generator: nhiễu → ảnh</text>\n')
    s += (f'<text x="490" y="40" text-anchor="middle" font-size="13.5" '
          f'fill="{ORANGE}" font-weight="600">Discriminator: ảnh → thật/giả</text>\n')

    # --- gradient lan ngược (tím, đứt nét) từ D về G ---
    gx = Dx + Dw + 38 + 56
    s += (f'<path d="M {gx} {Dmid + 32} L {gx} 312 L 188 312 L 188 {y_top + bh + 9}" '
          f'fill="none" stroke="{PURPLE}" stroke-width="2.2" stroke-dasharray="6,5" '
          f'marker-end="url(#arr-p)"/>\n')
    s += (f'<text x="402" y="307" text-anchor="middle" font-size="13" '
          f'fill="{PURPLE}" font-weight="600">gradient lan ngược — G chỉ học qua D</text>\n')

    s += "</svg>\n"
    (OUT / "gan-architecture.svg").write_text(s, encoding="utf-8")
    print("Saved gan-architecture.svg")


def main():
    make_vae()
    make_diffusion()
    make_sd()
    make_gan()


if __name__ == "__main__":
    main()
