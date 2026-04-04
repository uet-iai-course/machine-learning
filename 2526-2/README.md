# Học máy — Học kì 2, năm học 2025–2026

**@Giảng viên:** Vui lòng cập nhật các file `lecture-XX.html` cho từng buổi học khi cần. Bạn có thể dùng các bài giảng hiện có làm tham khảo.

## Nội dung học kì

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

## Chạy slide

Thư mục này chứa slide bài giảng viết bằng Reveal.js, có thể serve qua GitHub Pages hoặc máy chủ web cục bộ.

### Chạy trên máy cục bộ (khuyến nghị khi phát triển)

Chạy cục bộ có nhiều ưu điểm: xem được **ghi chú của giảng viên** và **tự động tải lại** khi có thay đổi.

1. Cài Node.js.
2. Vào thư mục này và chạy `npm install` để cài dependencies.
3. Chạy `npm start` để khởi động server.
4. Mở trình duyệt và truy cập http://localhost:8000.

Hoặc dùng server tích hợp của Python (không cần cài thêm gì):

```bash
python3 -m http.server 8765
# Mở http://localhost:8765
```

### Serve qua GitHub Pages

Không cần cấu hình gì thêm. Chỉ cần push thay đổi lên nhánh `main`, GitHub Pages sẽ tự động serve nội dung.

## Tạo lại hình minh hoạ

Mỗi bài giảng có thư mục `img/lecXX/scripts/` chứa các Python script sinh hình SVG. Để chạy:

```bash
cd img/lec07/scripts
python NN_ten_script.py
```

Yêu cầu: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.
