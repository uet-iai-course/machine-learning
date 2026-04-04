# Học máy — Trường Đại học Công nghệ, ĐHQGHN

Tài liệu bài giảng và bài tập về nhà cho môn **Học máy** tại Viện Trí tuệ Nhân tạo (IAI), Trường Đại học Công nghệ (UET), ĐHQGHN.

Slide bài giảng được xây dựng bằng [Reveal.js](https://revealjs.com/). Các hình minh hoạ được tạo tự động bằng Python (matplotlib, scipy, scikit-learn).

## Các năm học

| Năm học | Học kì | Thư mục |
|---------|--------|---------|
| 2025–2026 | 2 | [`2526-2/`](2526-2/) |

> Mỗi năm học mới sẽ được thêm vào dưới dạng một thư mục riêng (ví dụ: `2627-1/`).

## Cấu trúc mỗi năm học

```
XXXX-X/
├── index.html                  # Trang chủ (lịch giảng dạy)
├── lecture-*.html              # Slide bài giảng (Reveal.js)
├── assignments/
│   ├── assignment-*.html       # Bài tập về nhà
│   └── img/                    # Hình minh hoạ cho bài tập
└── img/
    └── lecXX/
        ├── *.svg               # Hình minh hoạ (generated)
        └── scripts/            # Python scripts tạo hình
```

## Thêm năm học mới

1. Tạo thư mục mới (ví dụ: `2627-1/`) bằng cách copy từ năm học gần nhất.
2. Cập nhật `index.html` ở root để thêm link đến năm học mới.
3. Xem hướng dẫn chi tiết trong `XXXX-X/README.md` của thư mục đó.

## Giấy phép

Dự án này được phát hành dưới [GNU General Public License v3](LICENSE).
