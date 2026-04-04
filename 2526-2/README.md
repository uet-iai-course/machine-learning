# README for 2526-2

**@Lecturers:** Please update the `lecture-XX.html` files for each lecture as needed. You can use the existing lectures as references.

## Nội dung học kì 2, 2025–2026

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

## Serving Lectures

This directory contains lecture slides written with Reveal.js that can be served via either GitHub Pages or a local web server.

### Serving Locally (Recommended for Development)

Serving the slides locally allows several advantages, such as the ability to see **speaker notes** and **automatic reloading** when you make changes to the slides.

1. Install Node.js.
2. Go to this directory in your terminal and run `npm install` to install dependencies.
3. Serve the presentation by running `npm start`.
4. Open your web browser and navigate to http://localhost:8000 to view the slides.

Alternatively, use Python's built-in server (no dependencies needed):

```bash
python3 -m http.server 8765
# Open http://localhost:8765
```

### Serving via GitHub Pages

Nothing special is needed. Just push your changes to the `main` branch, and GitHub Pages will automatically serve the content.

## Tạo lại hình minh hoạ

Mỗi bài giảng có thư mục `img/lecXX/scripts/` chứa các Python script. Để chạy:

```bash
cd img/lec07/scripts
python NN_script_name.py
```

Yêu cầu: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.
