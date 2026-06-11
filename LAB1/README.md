# Báo Cáo Thực Hành Lab 1
**Chủ đề: Thu thập và xử lý dữ liệu cho bài toán Face Detection**

## Mục tiêu bài Lab
1. Xây dựng bộ dữ liệu (dataset) hình ảnh khuôn mặt được gán nhãn (bounding box) để phục vụ cho việc huấn luyện mô hình AI phát hiện khuôn mặt (Face Detection).
2. Thu thập dữ liệu thực tế với độ đa dạng cao về ánh sáng, góc độ và đối tượng.
3. Sử dụng thư viện OpenCV và thuật toán Haar Cascade để tự động hóa quy trình gán nhãn.
4. Hiểu rõ cấu trúc lưu trữ dataset chuẩn.

## Cấu trúc thư mục Dataset
Bộ dữ liệu cần được đặt đúng cấu trúc để script xử lý chính xác:
```
face_detection_dataset/
├── images/     # Nơi chứa các ảnh đầu vào (.jpg, .png...)
├── labels/     # Nơi tự động sinh ra các file label (.txt)
└── logs/       # Nơi lưu file log cảnh báo ảnh không tìm thấy khuôn mặt
```
*Lưu ý: Ảnh đầu vào nên đặt tên theo quy tắc `TenNguoi_SoThuTu.jpg` (Ví dụ: `Khang_01.jpg`)*

## Hướng dẫn sử dụng các Script

Trong folder này có 2 script chính được trích xuất từ báo cáo:

### 1. Script cơ bản (`02_auto_label_faces.py`)
Sử dụng mô hình Haar Cascade mặc định `haarcascade_frontalface_default.xml` để nhận diện khuôn mặt chính diện.
- **Cách chạy:**
  ```bash
  python 02_auto_label_faces.py
  ```
- **Hạn chế:** Chỉ nhận diện tốt các góc mặt thẳng, ánh sáng đều. Không hiệu quả với các ảnh chụp góc nghiêng hoặc lóa sáng.

### 2. Script nâng cao (`02_auto_label_faces_advanced.py`)
Phiên bản cải tiến này khắc phục nhược điểm của phiên bản cơ bản thông qua "Chiến thuật 3 bước":
1. **Bước 1:** Quét chính diện (Frontal view).
2. **Bước 2:** Nếu không thấy, quét góc nghiêng (Profile view) bằng `haarcascade_profileface.xml`.
3. **Bước 3:** Nếu vẫn không thấy, lật ngược ảnh (Flip) và quét nghiêng lần nữa để bắt được các hướng nghiêng bên kia.

- **Cách chạy:**
  ```bash
  python 02_auto_label_faces_advanced.py
  ```

## Yêu cầu môi trường
- Python 3.x
- OpenCV (`pip install opencv-python`)
- Môi trường nên chạy trên Raspberry Pi 4 8GB hoặc máy tính cá nhân.

## Kết luận & Đề xuất (Từ báo cáo)
- Việc dùng điện thoại thu thập ảnh mang lại chất lượng tốt hơn webcam máy tính.
- Để khắc phục hoàn toàn nhược điểm góc nghiêng hoặc bị che khuất, có thể chuyển sang sử dụng các mô hình Deep Learning như MTCNN hoặc YOLO.
