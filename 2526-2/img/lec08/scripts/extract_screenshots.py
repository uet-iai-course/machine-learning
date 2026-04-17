"""Extract specific regions from PDF slides as screenshots."""
import fitz  # pymupdf

PDF_PATH = "/Users/tuanphong/teaching/UET/2526-2/MachineLearning/materials/Lectures/11_slides.pdf"
OUT_DIR = "../raw-screenshots/"

doc = fitz.open(PDF_PATH)

# Pages to extract (0-indexed) and their crop regions (relative fractions)
# Format: (page_0indexed, left%, top%, right%, bottom%, filename)
extracts = [
    # p.8 "From Tree to Regions and Back" — full content area
    (7, 0.03, 0.10, 0.97, 0.92, "p08-tree-regions-3d.png"),
    # p.17 "Heart — unpruned tree" — the two trees
    (16, 0.0, 0.10, 1.0, 0.92, "p17-heart-unpruned.png"),
    # p.24 "Wisdom of Crowds" — the plot
    (23, 0.35, 0.20, 0.95, 0.95, "p24-wisdom-crowds.png"),
    # p.29 "Choosing m" — the cancer plot
    (28, 0.48, 0.10, 0.98, 0.92, "p29-choosing-m.png"),
    # p.32 "Boosting" — the comparison plot
    (31, 0.05, 0.28, 0.55, 0.92, "p32-boosting-curves.png"),
    # p.35 "Simulated Data" — bagged vs boosted plots
    (34, 0.30, 0.30, 0.98, 0.90, "p35-simulated-data.png"),
]

for page_idx, l, t, r, b, fname in extracts:
    page = doc[page_idx]
    rect = page.rect
    # Convert relative fractions to absolute coordinates
    clip = fitz.Rect(
        rect.x0 + l * rect.width,
        rect.y0 + t * rect.height,
        rect.x0 + r * rect.width,
        rect.y0 + b * rect.height,
    )
    mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for high quality
    pix = page.get_pixmap(matrix=mat, clip=clip)
    pix.save(OUT_DIR + fname)
    print(f"Saved {fname} ({pix.width}x{pix.height})")

doc.close()
print("Done extracting screenshots.")
