# CLAUDE.md — Hướng dẫn cho Claude Code

Repo này chứa slide bài giảng môn **Học máy** tại Viện Trí tuệ Nhân tạo, Trường Đại học Công nghệ, ĐHQGHN. Slide viết bằng [Reveal.js](https://revealjs.com/), trình bày tiếng Việt cho sinh viên năm 2-3.

> **Convention thiết kế / nội dung slide**: đọc [`SLIDE_STYLE_GUIDE.md`](./SLIDE_STYLE_GUIDE.md). File CLAUDE.md này chỉ chứa quy tắc làm việc với repo (workflow, git, "đừng làm gì") + bài học từ session trước.

## Cấu trúc repo

```
.
├── index.html              # Trang chủ liệt kê các năm học
├── index-pages.css         # CSS cho các trang index
├── README.md
├── CLAUDE.md               # File này
├── SLIDE_STYLE_GUIDE.md    # Tiêu chuẩn thiết kế slide (đọc nếu sửa nội dung)
└── XXXX-X/                 # Mỗi học kì 1 thư mục, ví dụ 2526-2 = HK2 2025-2026
    ├── README.md
    ├── index.html
    ├── lecture-*.html      # Slide từng bài
    ├── lecture-style.css
    ├── assignments/        # Bài tập về nhà
    ├── img/lecXX/          # Hình ảnh
    │   └── scripts/        # Script Python sinh hình (matplotlib SVG)
    ├── qti_exports/        # Quiz QTI cho Canvas (gitignored)
    ├── plugin/, revealjs/
    └── package.json
```

PDF gốc các bài giảng ở `../../materials/Lectures/` (ví dụ `08_slides.pdf`).

## Run local

Server `slides` thường đã chạy sẵn trong **Claude Preview** trên port `8765` —
gọi `mcp__Claude_Preview__preview_list` để lấy `serverId`. Đừng spawn server mới.

Nếu cần chạy tay:
```bash
cd 2526-2
.conda/bin/python -m http.server 8765
```

GitHub Pages auto-deploy khi push lên `main`.

## Quy tắc số 1 — Preview trước khi nói "xong"

**Không bao giờ kết luận về layout chỉ bằng cách đọc code.**

Mọi thay đổi liên quan đến hiển thị (kích thước hình, layout, font, padding) phải verify bằng `preview_eval` + `preview_screenshot` (hoặc Chrome MCP) trước khi báo "đã sửa".

### Phát hiện tràn slide (Reveal.js)

**SAI** — kiểm tra này KHÔNG đủ:
```js
slide.scrollHeight === slide.clientHeight  // chỉ bắt scrollable overflow
```

Reveal có chiều cao logic cố định **700px** (xem `Reveal.getConfig().height`). Nội dung vượt 700 sẽ **đè lên `.footer`** mà KHÔNG kích hoạt scroll → `scrollHeight` không đổi.

**ĐÚNG**:
```js
const s = Reveal.getCurrentSlide();
const cfgH = Reveal.getConfig().height;     // 700
const fits = s.offsetHeight <= cfgH;
const overflowPx = s.offsetHeight - cfgH;   // > 0 = che footer
```

Hoặc so `slide.getBoundingClientRect().bottom` với `.footer.getBoundingClientRect().top`.

Sau khi đo, **luôn chụp screenshot** để mắt thường xác nhận footer hiện đầy đủ.

### Reveal indexing

- URL hash `#/h/v` và `Reveal.slide(h, v)` đều **0-indexed**
- Indicator góc dưới phải hiện h.v **1-indexed** — đừng nhầm
- URL hash dễ race với init Reveal — dùng `Reveal.slide()` trong `preview_eval` chắc hơn

### Cảnh báo: scroll view

Nếu `Reveal.isScrollView()` trả về `true`, `Reveal.slide(h, v)` sẽ **bị bỏ qua silent**
— indices không đổi, không lỗi. Trong scroll view, slides hiển thị như trang dài
liên tục.

**Cách phát hiện slide cụ thể trong scroll view:**
```js
const target = Array.from(document.querySelectorAll('.reveal section'))
  .find(s => s.querySelector('h2')?.textContent.includes('Lề cứng'));
const fits = target.offsetHeight <= 700;     // vẫn so với cfgH
target.scrollIntoView({ block: 'start' });
```

`offsetHeight` của từng `<section>` vẫn đúng để check fits với `cfgH = 700`
trong scroll view.

## Git workflow

- **Branch convention**: `lec{NN}-refine` cho refine bài N (ví dụ `lec08-refine`),
  `lec{NN}-draft` cho draft mới. Tên branch phải khớp **đúng số bài** — đừng tạo
  `lec09-refine` rồi nhồi nội dung lec08.
- **Đừng tự push** — chỉ push khi user yêu cầu rõ ràng.
- **Commit message format**:
  ```
  {Tóm tắt ngắn imperative dưới 70 ký tự}

  - Bullet list mô tả thay đổi (tiếng Việt OK).
  - …

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- PR base: `main`. Tiêu đề PR ngắn (<70 ký tự), body có Summary + Test plan.
- Trước khi commit: verify `<section>` / `</section>` tag balance bằng `grep -c`.

## Quy tắc tuyệt đối — "ĐỪNG"

Đây là quy tắc về **cách làm việc**. Quy tắc về **nội dung slide** xem [`SLIDE_STYLE_GUIDE.md`](./SLIDE_STYLE_GUIDE.md).

1. ❌ **Đừng tuyên bố "xong"** trước khi preview + screenshot xác nhận.
2. ❌ **Đừng tự push** lên remote — chỉ commit local. User sẽ ra lệnh push.
3. ❌ **Đừng tạo file documentation/README mới** trừ khi user yêu cầu.
4. ❌ **Đừng skip pre-commit hooks** (`--no-verify`).
5. ❌ **Đừng amend commit cũ** — luôn tạo commit mới. Chỉ amend khi user yêu cầu rõ.
6. ❌ **Đừng tạo slide mới** trừ khi user yêu cầu rõ. Sửa slide hiện có ưu tiên hơn thêm slide.
7. ❌ **Đừng over-annotate hình** (verdict box + legend + tag chồng nhau khi tiêu đề panel + caption đã đủ).

## Khi không chắc

- Hỏi user trước khi: thêm slide mới, thay đổi structure lớn, đụng vào content khác slide đang sửa.
- User-pause = dừng commit/push, đợi instruction tiếp.
- Auto mode = thực hiện ngay, ít interrupt.
- Khi user nói "xấu" / "không ổn" mà không nói cụ thể: phân tích hiện trạng (font, layout, balance, visual unity) rồi đề xuất 2-3 phương án để user chọn.

## Quy ước dịch thuật

Glossary chuẩn của môn nằm ở [`glossaries.yaml`](./glossaries.yaml) — copy từ
`iai-question-bank/docs/dev/glossaries/hoc-may.yml`. Đọc trước khi dịch
thuật ngữ mới. Schema: mỗi entry có `en`, `vi`, optional `abbrev`, `aliases`,
`notes`.

**Quy tắc chung khi dùng:**
- Tiếng Việt là chính; giữ tiếng Anh trong ngoặc ở **lần giới thiệu đầu tiên**
  (TOC + tiêu đề slide chính của khái niệm), các lần sau dùng tiếng Việt thuần.
- Khi cần override cho slide cụ thể, ghi rõ trong code comment hoặc cập nhật
  `notes` của entry trong `glossaries.yaml`.

## Python & sinh hình

Dùng `.conda/bin/python` (KHÔNG phải `python3` hệ thống):
```bash
.conda/bin/python 2526-2/img/lec09/scripts/07_mmc_example.py
```

Script SVG dùng `_common.py` chia sẻ trong cùng `scripts/` (palette màu, helpers).

## Cache-busting SVG

Sau mỗi lần regenerate SVG, **bump query string** trên `<img src>`:
```html
<img src="img/lec09/mmc-example.svg?v=4" ... />
```
Trình duyệt cache theo URL. Không bump → preview vẫn hiện ảnh cũ.

## Mẹo CSS hay quên

- **Khoảng trắng dưới `<img>`**: bao bằng wrapper với `font-size:0;line-height:0;` để bỏ baseline descender (~50px nếu không xử lý → đủ làm tràn slide).
- **Reveal `.footer`** phải nằm trong `.slides` (anh em với `<section>`), không phải anh em với `.slides`.
- **Inline style** chấp nhận được cho slide đặc thù; pattern lặp lại nên đưa vào `lecture-style.css`.
- **Grid `1fr` cells unequal**: nếu cell có MathJax formula (hoặc bất kỳ content `min-content` lớn), `grid-template-columns: 1fr 1fr 1fr` sẽ ép cell đó phình theo formula, làm tổng vượt slide width. Fix: `minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)` + `min-width: 0` trên mỗi grid item.
- **MathJax wrap formula**: cả `\[...\]` (display) và `\(...\)` (inline) đều wrap ở binary operators khi cell hẹp. Fix chắc chắn: wrapper với `white-space: nowrap` + giảm `font-size` để fit.
- **Display math `\[...\]` cũng tràn ngang container**: không wrap ở operator nhưng vẫn extend horizontally (scrollWidth > clientWidth). Phải đo bằng `formulaContainer.scrollWidth - clientWidth`. Nếu tràn: dùng notation compact (`e^{...}` thay `\exp(...)`, `\tfrac` thay `\frac`) hoặc tăng width container.
- **KaTeX trong heading bị viết hoa**: Reveal `white` theme set `text-transform: uppercase` cho h1/h2/h3 → Greek letters (\alpha, \gamma) bị render thành Latin uppercase. Fix căn cơ đã có trong `lecture-style.css`:
  ```css
  .reveal .katex, .reveal .katex *,
  .reveal mjx-container, .reveal mjx-container *,
  .reveal .MathJax, .reveal .MathJax * { text-transform: none !important; }
  ```
  Không cần workaround per-slide. Khi sửa CSS này nhớ bump `?v=N` trên link để browser fetch.

## matplotlib mẹo

- **3D subplot bị cắt rìa**: `tight_layout` không tính đúng z-axis label. Dùng `subplots_adjust(left=..., right=...)` thủ công + `savefig(bbox_inches="tight", pad_inches=0.3)` thay vì để matplotlib tự crop.
- Sau khi sửa script SVG, **luôn bump `?v=N`** trong `<img src>` của slide HTML, kẻo browser dùng cache cũ.

## Đọc thêm

- [`SLIDE_STYLE_GUIDE.md`](./SLIDE_STYLE_GUIDE.md) — tiêu chuẩn thiết kế slide (color, pattern, wording, badge `📖 Tự học`…).
- [`glossaries.yaml`](./glossaries.yaml) — glossary thuật ngữ ML chuẩn (copy từ `iai-question-bank/docs/dev/glossaries/hoc-may.yml`).
- `2526-2/README.md` — đặc thù của học kì cụ thể.
