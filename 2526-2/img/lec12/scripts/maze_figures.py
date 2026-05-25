"""Maze figures for lec12 (RL).

Sinh 3 file SVG dùng cho Bài 12:
- maze.svg:        bản đồ + Start/Goal (slide 3.2)
- maze-value.svg:  bản đồ + giá trị V(s) (heatmap đỏ→xanh) + số (slide 3.4)
- maze-policy.svg: bản đồ + mũi tên π*(s) (slide 3.6)

Cả 3 dùng cùng một maze layout — sinh viên dễ so sánh.

Cải tiến so với hình PNG gốc (David Silver):
- Palette nhất quán với slide (xanh tác tử, cam goal, đỏ phần thưởng)
- Heatmap giá trị: đỏ (xa goal) → xanh lá (gần goal) — trực giác hơn số trần
- Mũi tên dày, có viền — dễ nhìn từ xa
- Start/Goal có nhãn S/G + viền màu — tự giải thích, không cần text bên ngoài
- Cell có grid line nhẹ — đỡ "trống trải"
"""
from pathlib import Path
from collections import deque

# ---------- Maze layout ----------
# 0 = open, 1 = wall, 2 = start, 3 = goal
MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 0
    [2, 0, 1, 1, 0, 1, 0, 0, 0],  # row 1: Start ở (1, 0)
    [0, 0, 0, 1, 0, 0, 0, 1, 0],  # row 2
    [0, 1, 0, 0, 1, 0, 1, 1, 0],  # row 3
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # row 4
    [0, 0, 0, 0, 0, 0, 1, 1, 3],  # row 5: Goal ở (5, 8)
]
ROWS = len(MAZE)
COLS = len(MAZE[0])

# ---------- Visual config ----------
SLOT = 66           # khoảng cách giữa các tâm ô (bao gồm gap)
GAP = 4             # gap giữa hai ô liền kề
CELL = SLOT - GAP   # kích thước thực của ô vẽ ra
PAD = 12
WIDTH = COLS * SLOT + 2 * PAD - GAP
HEIGHT = ROWS * SLOT + 2 * PAD - GAP

# Palette (đồng bộ với lecture-style.css)
WALL = "#1a2332"
OPEN = "#ffffff"
GRID = "#cfd6dc"
START_FILL = "#bfe2eb"
START_EDGE = "#1E93AB"
GOAL_FILL = "#fde2c4"
GOAL_EDGE = "#e8732a"
ARROW = "#E62727"
TEXT = "#1a2332"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, "
        "Arial, sans-serif")


def find(val: int) -> tuple[int, int]:
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == val:
                return r, c
    raise ValueError(f"value {val} not in maze")


START = find(2)
GOAL = find(3)


def is_open(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS and MAZE[r][c] != 1


def compute_distances() -> list[list[int | None]]:
    """BFS từ goal — trả về ma trận khoảng cách (số bước đến goal)."""
    dist: list[list[int | None]] = [[None] * COLS for _ in range(ROWS)]
    dist[GOAL[0]][GOAL[1]] = 0
    queue: deque[tuple[int, int]] = deque([GOAL])
    while queue:
        r, c = queue.popleft()
        cur = dist[r][c]
        assert cur is not None
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if is_open(nr, nc) and dist[nr][nc] is None:
                dist[nr][nc] = cur + 1
                queue.append((nr, nc))
    return dist


def cell_xy(r: int, c: int) -> tuple[float, float]:
    """Top-left pixel of cell (r, c)."""
    return PAD + c * SLOT, PAD + r * SLOT


def cell_center(r: int, c: int) -> tuple[float, float]:
    x, y = cell_xy(r, c)
    return x + CELL / 2, y + CELL / 2


def value_color(v: int | None, vmax: int) -> str:
    """Heatmap: v=0 (at goal) → xanh lá; v=-vmax (xa nhất) → đỏ."""
    if v is None:
        return OPEN
    # normalize: t = 1 ở goal, 0 ở xa nhất
    t = 1.0 - abs(v) / max(vmax, 1)
    # Red → yellow → green
    if t > 0.5:
        ratio = (t - 0.5) * 2
        r = int(255 - 100 * ratio)
        g = 220
        b = int(120 + 60 * ratio)
    else:
        ratio = t * 2
        r = 245
        g = int(150 + 70 * ratio)
        b = 120
    return f"rgb({r},{g},{b})"


# ---------- SVG fragments ----------

def render_cells(fill_override: dict[tuple[int, int], str] | None = None) -> list[str]:
    """Vẽ tất cả ô. fill_override = mapping (r,c) → màu fill cho value heatmap."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            x, y = cell_xy(r, c)
            v = MAZE[r][c]
            if v == 1:
                fill, stroke, sw = WALL, WALL, 1
            elif v == 2:
                fill, stroke, sw = START_FILL, START_EDGE, 2.5
            elif v == 3:
                fill, stroke, sw = GOAL_FILL, GOAL_EDGE, 2.5
            else:
                fill, stroke, sw = OPEN, GRID, 1
            if fill_override and (r, c) in fill_override and v not in (1, 2, 3):
                fill = fill_override[(r, c)]
                sw = 1
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL}" height="{CELL}" '
                f'rx="3" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
    return out


def render_start_goal_labels(start_corner: bool = False) -> list[str]:
    """Nhãn S, G ở trong ô start/goal. start_corner=True đặt S ở góc (slide policy)."""
    gx, gy = cell_center(*GOAL)
    out = [
        f'<text x="{gx:.1f}" y="{gy + 8:.1f}" text-anchor="middle" '
        f'font-size="26" font-weight="700" fill="{GOAL_EDGE}">G</text>',
    ]
    if start_corner:
        sx, sy = cell_xy(*START)
        out.insert(0,
            f'<text x="{sx + 7:.1f}" y="{sy + 18:.1f}" '
            f'font-size="15" font-weight="700" fill="{START_EDGE}">S</text>')
    else:
        sx, sy = cell_center(*START)
        out.insert(0,
            f'<text x="{sx:.1f}" y="{sy + 8:.1f}" text-anchor="middle" '
            f'font-size="26" font-weight="700" fill="{START_EDGE}">S</text>')
    return out


def render_values(dist: list[list[int | None]]) -> list[str]:
    """Số -k ở mỗi ô open (trừ start/goal đã có chữ)."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            d = dist[r][c]
            if d is None or MAZE[r][c] in (2, 3):
                continue
            cx, cy = cell_center(r, c)
            out.append(
                f'<text x="{cx:.1f}" y="{cy + 6:.1f}" text-anchor="middle" '
                f'font-size="19" font-weight="600" fill="{TEXT}">−{d}</text>'
            )
    # Start và Goal cũng cần hiển thị giá trị
    sd = dist[START[0]][START[1]]
    gd = dist[GOAL[0]][GOAL[1]]
    sx, sy = cell_center(*START)
    gx, gy = cell_center(*GOAL)
    if sd is not None and sd > 0:
        out.append(
            f'<text x="{sx:.1f}" y="{sy - 8:.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="{START_EDGE}">S</text>'
        )
        out.append(
            f'<text x="{sx:.1f}" y="{sy + 14:.1f}" text-anchor="middle" '
            f'font-size="18" font-weight="700" fill="{TEXT}">−{sd}</text>'
        )
    if gd == 0:
        out.append(
            f'<text x="{gx:.1f}" y="{gy - 8:.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="{GOAL_EDGE}">G</text>'
        )
        out.append(
            f'<text x="{gx:.1f}" y="{gy + 14:.1f}" text-anchor="middle" '
            f'font-size="18" font-weight="700" fill="{TEXT}">−0</text>'
        )
    return out


def render_arrow(r: int, c: int, dr: int, dc: int) -> str:
    """Vẽ mũi tên 1 polygon (shaft + head) — không dùng marker để tránh
    'đầu tròn lòi ra' khi marker overlap với line stroke."""
    cx, cy = cell_center(r, c)
    L = CELL * 0.34          # nửa độ dài tổng (tip cách tâm L)
    BACK = L * 0.6           # đuôi cách tâm (ngắn hơn tip để cân thị giác)
    SHAFT_W = 2.0            # nửa bề dày shaft
    HEAD_W = 6.5             # nửa bề rộng đáy đầu mũi tên
    HEAD_LEN = 8.5           # chiều dài đầu mũi tên (tính từ tip)

    # Hướng & pháp tuyến
    tip_x, tip_y = cx + dc * L, cy + dr * L
    back_x, back_y = cx - dc * BACK, cy - dr * BACK
    head_base_x = tip_x - dc * HEAD_LEN
    head_base_y = tip_y - dr * HEAD_LEN
    # Vector vuông góc (xoay 90°)
    px, py = -dr, dc

    pts = [
        (back_x + px * SHAFT_W, back_y + py * SHAFT_W),       # đuôi-trái
        (head_base_x + px * SHAFT_W, head_base_y + py * SHAFT_W),  # cổ đầu-trái
        (head_base_x + px * HEAD_W, head_base_y + py * HEAD_W),    # cánh đầu-trái
        (tip_x, tip_y),                                            # đỉnh tip
        (head_base_x - px * HEAD_W, head_base_y - py * HEAD_W),    # cánh đầu-phải
        (head_base_x - px * SHAFT_W, head_base_y - py * SHAFT_W),  # cổ đầu-phải
        (back_x - px * SHAFT_W, back_y - py * SHAFT_W),       # đuôi-phải
    ]
    pts_str = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    return f'<polygon points="{pts_str}" fill="{ARROW}"/>'


def best_action(r: int, c: int, dist: list[list[int | None]]) -> tuple[int, int] | None:
    """Hướng tốt nhất từ (r,c) — đến neighbor có khoảng cách nhỏ nhất."""
    if dist[r][c] is None:
        return None
    if MAZE[r][c] == 3:  # goal — không cần arrow
        return None
    best: tuple[int, int] | None = None
    best_d: int | None = dist[r][c]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_open(nr, nc) and dist[nr][nc] is not None:
            nd = dist[nr][nc]
            assert nd is not None and best_d is not None
            if nd < best_d:
                best_d = nd
                best = (dr, dc)
    return best


def render_policy(dist: list[list[int | None]]) -> list[str]:
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            d = best_action(r, c, dist)
            if d is not None:
                out.append(render_arrow(r, c, *d))
    return out


def svg_header(extra_defs: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'font-family="{FONT}">\n'
        f'{extra_defs}'
    )


# ---------- Generate 3 SVGs ----------

def make_maze() -> str:
    parts = [svg_header()]
    parts.extend('  ' + s for s in render_cells())
    parts.extend('  ' + s for s in render_start_goal_labels())
    parts.append('</svg>\n')
    return '\n'.join(parts)


def make_value() -> str:
    dist = compute_distances()
    vmax = max(d for row in dist for d in row if d is not None)
    # Heatmap fill per open cell (NOT start/goal — they keep their fill)
    heatmap: dict[tuple[int, int], str] = {}
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == 0 and dist[r][c] is not None:
                heatmap[(r, c)] = value_color(-dist[r][c], vmax)
    parts = [svg_header()]
    parts.extend('  ' + s for s in render_cells(fill_override=heatmap))
    parts.extend('  ' + s for s in render_values(dist))
    parts.append('</svg>\n')
    return '\n'.join(parts)


def make_policy() -> str:
    dist = compute_distances()
    parts = [svg_header()]
    parts.extend('  ' + s for s in render_cells())
    parts.extend('  ' + s for s in render_start_goal_labels(start_corner=True))
    parts.extend('  ' + s for s in render_policy(dist))
    parts.append('</svg>\n')
    return '\n'.join(parts)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent
    files = {
        "maze.svg": make_maze(),
        "maze-value.svg": make_value(),
        "maze-policy.svg": make_policy(),
    }
    for name, content in files.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
