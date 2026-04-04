# Học máy — Học kì 2, năm học 2025-2026

Tài liệu bài giảng và bài tập về nhà cho môn **Học máy** tại Viện Trí tuệ Nhân tạo (IAI), Trường Đại học Công nghệ, ĐHQGHN.

## Nội dung

Slide bài giảng được xây dựng bằng [Reveal.js](https://revealjs.com/). Các hình minh hoạ được tạo tự động bằng Python (matplotlib, scipy, scikit-learn).

| Tuần | Bài giảng | Bài tập |
|------|-----------|---------|
| 1 | Tổng quan học phần & Giới thiệu học máy | #1 |
| 2 | Hồi quy tuyến tính | #2 |
| 3 | Bài toán phân lớp | #3 |
| 4 | Năng lực tổng quát hoá & Lựa chọn mô hình | #4 |
| 5 | Mô hình phi tuyến | #5 |
| 6 | Giảm chiều dữ liệu | #6 |
| 7 | Bài toán phân cụm | #7 |
| 8 | Mô hình cây quyết định | — |
| 9 | Máy vectơ hỗ trợ (SVM) | — |
| 10 | Mạng nơ-ron nhân tạo | — |

## Cấu trúc thư mục

```
2526-2/
├── index.html                  # Trang chủ (lịch giảng dạy)
├── lecture-*.html              # Slide bài giảng (Reveal.js)
├── assignments/
│   ├── assignment-*.html       # Bài tập về nhà
│   └── img/                    # Hình minh hoạ cho bài tập
└── img/
    └── lec*/
        ├── *.svg               # Hình minh hoạ (generated)
        └── scripts/            # Python scripts tạo hình
```

## Chạy local

```bash
cd 2526-2
python3 -m http.server 8765
# Mở http://localhost:8765 trong trình duyệt
```

## Tạo lại hình minh hoạ

Mỗi bài giảng có thư mục `img/lecXX/scripts/` chứa các Python script. Để chạy:

```bash
cd 2526-2/img/lec07/scripts
python NN_script_name.py
```

Yêu cầu: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.
