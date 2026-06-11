# Báo Cáo Thực Hành Lab 3
**Chủ đề: Lượng tử hóa INT8 và chạy Face Detection YOLOv8 trên Raspberry Pi**

## Mục tiêu bài Lab
1. Triển khai mô hình nhận diện khuôn mặt (Face Detection) dựa trên kiến trúc YOLO trên nền tảng Raspberry Pi.
2. Thực hiện tối ưu hóa mô hình bằng kỹ thuật lượng tử hóa (Quantization - INT8) để giảm kích thước và tăng tốc độ xử lý trên phần cứng nhúng.
3. Đánh giá hiệu năng thực tế thông qua các chỉ số: Latency (Độ trễ) và FPS (Khung hình trên giây) để so sánh giữa bản Float32 và bản INT8.

## Chuẩn bị môi trường & Export Mô hình
- Cài đặt `ultralytics`, `ai-edge-litert` (hoặc `tflite-runtime`), `opencv-python`.
- Để chuyển đổi mô hình (PTQ - Post Training Quantization) sang INT8 trên PC:
  ```bash
  yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True data=dataset_yolo/data.yaml
  ```
- Kết quả nhận được bao gồm 2 file chính để test: `best_float32.tflite` và `best_full_integer_quant.tflite` (hoặc `best_int8.tflite`).

## Hướng dẫn các script

Trong folder này có 4 file script trích xuất từ báo cáo:

1. **`01_check_model_int8.py`**: 
   Dùng để kiểm tra nhanh file TFLite. Báo cáo chi tiết Input/Output tensor type và tham số quantization (scale, zero-point). 

2. **`02_detect_pc_ultralytics.py`**: 
   Chạy dự đoán trên máy tính (PC) sử dụng trực tiếp thư viện `ultralytics`. Pipeline tiền/hậu xử lý được thực hiện tự động.

3. **`03_detect_pi_tflite.py`**: 
   Script chuyên dụng để chạy trên Raspberry Pi bằng thư viện `tflite_runtime` (hoặc `ai_edge_litert`). Pipeline nhận diện phải viết thủ công: chuẩn hóa kích thước, lượng tử hóa đầu vào (nếu là INT8), gọi Interpreter, giải lượng tử hóa, tính toán lại Bounding Box và áp dụng NMS (Non-Maximum Suppression). 

4. **`04_benchmark_tflite.py`**: 
   Dùng để benchmark đo tốc độ (FPS, Latency, P50, P90, P99) mô hình TFLite (Float32 hoặc INT8) qua 100 lần lặp. Có thể chạy trên PC hoặc Pi để so sánh. 

## Tổng hợp kết quả thực nghiệm

1. **Kích thước mô hình:**
   - INT8 nhỏ hơn Float32 khoảng **3.7 lần** (do chuyển từ 32-bit xuống 8-bit).
2. **Trên Máy Tính (PC - x86_64):**
   - **Tốc độ:** INT8 đạt ~39 FPS (25.6 ms), Float32 đạt ~13 FPS (75 ms) -> Nhanh hơn gần **3 lần**. (PC hỗ trợ rất tốt tập lệnh SIMD/AVX cho số nguyên).
   - **Độ chính xác:** Do đây là Post-Training Quantization (PTQ) chưa áp dụng QAT, độ tin cậy (Confidence) của INT8 giảm đáng kể (từ 0.95 xuống khoảng 0.50).
3. **Trên Raspberry Pi (ARM):**
   - **Tốc độ:** INT8 đạt ~1.55 FPS (645 ms), Float32 đạt ~1.11 FPS (902 ms) -> Tốc độ chỉ tăng **1.4 lần**. Nguyên nhân do kiến trúc ARM hỗ trợ SIMD hạn chế hơn, đồng thời CPU phải gánh thêm nhiều chi phí overhead từ Python và xử lý tiền/hậu xử lý ảnh.
   - **Độ chính xác:** Tương tự như trên PC, phiên bản lượng tử hóa có sự sụt giảm độ tin cậy. Tuy nhiên, Float32 trên Pi (0.93) cũng hơi lệch nhẹ so với PC (0.95) do quy trình tiền/hậu xử lý tự viết thủ công bằng OpenCV không hoàn toàn tương đồng 100% với hàm ẩn bên trong framework Ultralytics.
