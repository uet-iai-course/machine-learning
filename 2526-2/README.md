# Học máy — Học kì 2, năm học 2025–2026

**@Giảng viên:** Vui lòng cập nhật các file `lecture-XX.html` cho từng buổi học khi cần. Bạn có thể dùng các bài giảng hiện có làm tham khảo.

## Nội dung học kì

| Tuần | Bài giảng | Bài tập |
|------|-----------|---------|
| 1 | Bài 1a: Tổng quan học phần; Bài 1b: Giới thiệu về học máy | #1 |
| 2 | Bài 2: Hồi quy tuyến tính | #2 |
| 3 | Bài 3: Bài toán phân lớp | #3 |
| 4 | Bài 4a: Năng lực tổng quát hoá của mô hình; Bài 4b: Lựa chọn mô hình | #4 |
| 5 | Bài 5: Mô hình phi tuyến | #5 |
| 6 | Bài 6: Giảm chiều dữ liệu | #6 |
| 7 | Bài 7: Bài toán phân cụm | #7 |
| 8 | Bài 8: Mô hình cây quyết định | *(chưa có bài tập)* |
| 9 | Bài 9: Máy vectơ hỗ trợ (SVM) | *(chưa có bài tập)* |
| 10 | Kiểm tra giữa kỳ | — |
| 11 | Bài 11: Giới thiệu về mạng nơ-ron | *(chưa có bài tập)* |
| 12 | Bài 12: Giới thiệu về học tăng cường | *(chưa có bài tập)* |
| 13 | Bài 13: Giới thiệu về mô hình sinh | *(chưa có bài tập)* |
| 14 | Bài 14: Học máy trong thực tế | *(chưa có bài tập)* |
| 15 | Kiểm tra cuối kì | — |

## Chạy slide

Thư mục này chứa slide bài giảng viết bằng Reveal.js, có thể serve qua GitHub Pages hoặc máy chủ web cục bộ.
Hai cách chạy dưới đây dùng hai cổng mặc định khác nhau theo cấu hình hiện tại: Node.js dùng `8000`, còn Python dùng `8765`.

### Cách 1: Node.js (hỗ trợ ghi chú giảng viên & tự động tải lại)

1. Cài [Node.js](https://nodejs.org/).
2. Trong thư mục `2526-2`, chạy `npm install` để cài dependencies.
3. Chạy `npm start` để khởi động server (port 8000).
4. Mở trình duyệt và truy cập http://localhost:8000.

### Cách 2: Python (không cần cài thêm gì)

```bash
# Nếu đang ở thư mục gốc của repo
cd 2526-2
python3 -m http.server 8765
# Mở http://localhost:8765
```

### Serve qua GitHub Pages

Không cần cấu hình gì thêm. Chỉ cần push thay đổi lên nhánh `main`, GitHub Pages sẽ tự động serve nội dung.

## Tạo lại hình minh hoạ

Mỗi bài giảng có thư mục `img/lecXX/scripts/` chứa các Python script để sinh lại hình minh hoạ. Ví dụ, để tạo lại toàn bộ hình cho bài 7:

```bash
# Chạy từ thư mục 2526-2
cd img/lec07/scripts
for f in *.py; do python3 "$f"; done
```

Hoặc chạy từng script riêng lẻ:

```bash
# Chạy từ thư mục 2526-2
cd img/lec07/scripts
python3 08_clustering_example.py
```

Yêu cầu: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.
