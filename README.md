# Học máy — Trường Đại học Công nghệ, ĐHQGHN

Tài liệu bài giảng và bài tập về nhà cho môn **Học máy** tại Viện Trí tuệ Nhân tạo (IAI), Trường Đại học Công nghệ (UET), ĐHQGHN.

Slide bài giảng được xây dựng bằng [Reveal.js](https://revealjs.com/). Nhiều hình minh hoạ được sinh tự động bằng Python (matplotlib, scipy, scikit-learn) và được lưu dưới dạng SVG hoặc PNG.

## Các năm học

| Năm học | Học kì | Thư mục |
|---------|--------|---------|
| 2025–2026 | 2 | [`2526-2/`](2526-2/) |

> Mỗi năm học mới sẽ được thêm vào dưới dạng một thư mục riêng (ví dụ: `2627-1/`).

## Cấu trúc mỗi năm học

> Cấu trúc dưới đây chỉ mang tính khái quát; một số thư mục có thể có thêm file cấu hình, tài nguyên hoặc script phụ trợ.

```
XXXX-X/
├── index.html                  # Trang chủ (lịch giảng dạy)
├── lecture-*.html              # Slide bài giảng (Reveal.js)
├── lecture-style.css           # CSS dùng chung cho các slide
├── assignments/
│   ├── assignment-*.html       # Bài tập về nhà
│   └── img/                    # Hình minh hoạ cho bài tập
└── img/
    └── lecXX/
        ├── *.svg / *.png       # Hình minh hoạ
        └── scripts/            # Python scripts tạo hình
```

## Thêm năm học mới

1. Tạo thư mục mới (ví dụ: `2627-1/`) bằng cách copy từ năm học gần nhất.
2. Cập nhật `index.html` ở root để thêm link đến năm học mới.
3. Xem hướng dẫn chi tiết trong `README.md` của thư mục năm học tương ứng.

## Giấy phép

Dự án này được phát hành dưới [GNU General Public License v3](LICENSE).
