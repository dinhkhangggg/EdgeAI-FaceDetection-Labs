# Báo Cáo Thực Hành Lab 2
**Chủ đề: Huấn luyện mô hình Face Detection bằng YOLOv8**

## Mục tiêu bài Lab
Huấn luyện mô hình AI (YOLOv8) để tự động phát hiện khuôn mặt từ bộ dữ liệu tự thu thập ở Lab 1. 

## Mô tả tập dữ liệu
Kế thừa bộ dữ liệu từ Lab 1 (đã gán nhãn và chuyển sang định dạng YOLO). Tổng số ảnh là 232 ảnh.

## Hướng dẫn các bước thực hiện

### Bước 1: Chuẩn bị dữ liệu
Chạy script `01_prepare_yolo_dataset.py` để chia tập dữ liệu thành Train/Val/Test với tỷ lệ tương ứng (70/15/15) và tự động tạo file cấu hình `data.yaml`.
```bash
python 01_prepare_yolo_dataset.py
```
**Kết quả chia tập:**
- Train: ~69.83%
- Val: ~14.65%
- Test: ~15.52%

### Bước 2: Huấn luyện mô hình (Training Process)
Dùng phiên bản YOLOv8 Nano (`yolov8n.pt`) vì kích thước bộ dữ liệu ở mức nhỏ, phù hợp triển khai sau này trên thiết bị Edge như Raspberry Pi 4 để tránh tràn VRAM.
Lệnh chạy huấn luyện trên GPU:
```bash
yolo detect train data=dataset_yolo/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16 device=0
```
*(Tham số siêu tham số: `imgsz=640`, `epochs=50`, `batch=16`)*

### Bước 3: Đánh giá mô hình
Kết quả sau 50 chu kỳ huấn luyện đạt chất lượng rất tốt:
- **Precision (P):** 96.9%
- **Recall (R):** 90.3%
- **mAP50:** 90.8%
- **mAP50-95:** 77.6%
Mô hình tổng quát hóa tốt, không bị Overfitting. Tỷ lệ phát hiện trúng khuôn mặt đạt 91% tổng số lượng khuôn mặt thực tế trong tập test.

### Bước 4: Dự đoán (Inference)
**Dự đoán trên thư mục ảnh Test:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset_yolo/images/test conf=0.25 save=True
```

**Dự đoán thời gian thực qua Webcam:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0 conf=0.25
```
*(Tốc độ xử lý của mô hình ước tính khoảng 3.9ms/ảnh).*

### Bước 5: Triển khai trên Edge Device (ONNX Export)
Để đảm bảo mô hình hoạt động ổn định trên Raspberry Pi hoặc các nền tảng giới hạn về thư viện, tiến hành xuất trọng số sang chuẩn ONNX (tương thích đa nền tảng):
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```
File `best.onnx` sẽ được tạo ra với dung lượng nhẹ (khoảng 11.7 MB).

## Hướng cải tiến
- Tăng cường dữ liệu (Data Augmentation), chụp ảnh ở các góc khuất và ánh sáng phức tạp.
- Sử dụng LabelImg để kiểm duyệt lại nhãn sinh tự động.
- Dùng `yolov8s.pt` kết hợp tăng số lượng epochs (VD: 100) để đánh giá.
