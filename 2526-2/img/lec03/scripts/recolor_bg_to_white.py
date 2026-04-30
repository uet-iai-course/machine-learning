"""Replace light-blue / blue-gray background in lec03 PNGs with white.

Cách dùng:
    .conda/bin/python 2526-2/img/lec03/scripts/recolor_bg_to_white.py

Logic:
1. Quét mọi `*.png` trong `2526-2/img/lec03/` (bỏ qua thư mục `scripts/`).
2. Lấy mẫu 4 góc → ước lượng màu nền.
3. Nếu nền là "sáng + xanh-xám" (mỗi kênh ∈ [210, 252], B ≥ G − 2 ≥ R − 4,
   max-min ≤ 30) thì:
   - Pass 1: flood-fill từ 4 góc với tolerance 30 → xử lý được nền gradient,
     không lan vào plot area (nội thất chart cách biệt với mép).
   - Pass 2: blanket-replace các pixel gần bg (Euclidean dist < 14) để dọn
     những patch nhỏ không nối với góc.
4. Bỏ qua ảnh có nền đen, trắng, hoặc bất kỳ màu nào không "sáng + xanh-xám"
   (ảnh người, MNIST, vòng đời ML, …).
5. Ghi đè in-place; dùng git để rollback nếu cần.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

LEC_DIR = Path(__file__).resolve().parent.parent
FLOOD_TOL = 30        # tolerance cho floodfill (mỗi kênh)
DIST_TOL = 14.0       # Euclidean distance cho blanket-replace pass


def sample_corners(arr: np.ndarray) -> np.ndarray:
    """Median RGB của 4 góc (patch 5×5)."""
    h, w, _ = arr.shape
    p = 5
    patches = [
        arr[:p, :p, :3], arr[:p, w - p:, :3],
        arr[h - p:, :p, :3], arr[h - p:, w - p:, :3],
    ]
    return np.array([np.median(c.reshape(-1, 3), axis=0) for c in patches])


def is_light_bluish(rgb: np.ndarray) -> bool:
    """True nếu rgb là 'sáng + thiên xanh-xám':
       - mỗi kênh ∈ [210, 254]
       - B ≥ G − 2 ≥ R − 4 (cho phép gradient nhẹ)
       - max-min ≤ 30 (không bão hoà)
    """
    r, g, b = rgb
    if not (210 <= r <= 254 and 210 <= g <= 254 and 210 <= b <= 254):
        return False
    if not (b >= g - 2 and g >= r - 4):
        return False
    return max(r, g, b) - min(r, g, b) <= 30


def estimate_bg(arr: np.ndarray) -> np.ndarray | None:
    corners = sample_corners(arr)
    matches = [is_light_bluish(c) for c in corners]
    if sum(matches) < 3:
        return None
    return np.median([c for c, m in zip(corners, matches) if m], axis=0)


def floodfill_from_corners(im: Image.Image) -> int:
    """Flood-fill 4 góc → trắng. Trả về số pixel đổi (xấp xỉ qua diff)."""
    w, h = im.size
    before = np.array(im).copy()
    seeds = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]
    for seed in seeds:
        ImageDraw.floodfill(im, seed, value=(255, 255, 255, 255),
                            thresh=FLOOD_TOL)
    after = np.array(im)
    return int(np.any(before != after, axis=-1).sum())


def blanket_replace(arr: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, int]:
    """Đổi pixel có Euclidean dist < DIST_TOL từ bg → trắng.
       Pass này dọn các patch nhỏ không nối với góc (vd. khoảng trống nhỏ
       bị đường kẻ chia cắt khỏi mép)."""
    rgb = arr[:, :, :3].astype(np.float32)
    dist = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    mask = dist < DIST_TOL
    out = arr.copy()
    out[mask, 0] = 255
    out[mask, 1] = 255
    out[mask, 2] = 255
    return out, int(mask.sum())


def process(path: Path) -> tuple[bool, str]:
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    bg = estimate_bg(arr)
    if bg is None:
        corners_avg = tuple(int(x) for x in sample_corners(arr).mean(axis=0))
        return False, f"skip (bg ≈ {corners_avg})"

    # Pass 1: floodfill từ 4 góc — xử lý được gradient
    n1 = floodfill_from_corners(im)
    arr = np.array(im)

    # Pass 2: blanket-replace cho patch nhỏ không nối góc
    arr, n2 = blanket_replace(arr, bg)
    Image.fromarray(arr, mode="RGBA").save(path, optimize=True)

    total = arr.shape[0] * arr.shape[1]
    pct = 100.0 * (n1 + n2) / total
    return True, (f"flood={n1:>9d}px, blanket={n2:>9d}px "
                  f"(~{pct:5.1f}%) — bg ≈ {tuple(int(x) for x in bg)}")


def main() -> None:
    pngs = sorted(LEC_DIR.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in {LEC_DIR}")
        sys.exit(1)
    print(f"Scanning {len(pngs)} PNGs in {LEC_DIR}...")
    n_ok = 0
    for p in pngs:
        changed, msg = process(p)
        prefix = "✓" if changed else "·"
        print(f"  {prefix} {p.name:55s} {msg}")
        if changed:
            n_ok += 1
    print(f"\n{n_ok}/{len(pngs)} PNGs recolored to white background.")


if __name__ == "__main__":
    main()
