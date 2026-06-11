# Báo Cáo Thực Hành Lab 6
**Chủ đề: Tối ưu Face Detection bằng Multithreading và Multiprocessing trên Raspberry Pi**

## Mục tiêu bài Lab
Triển khai nhận diện khuôn mặt bằng YOLOv8 với mô hình TFLite INT8 trên mạch Raspberry Pi. Đồng thời đo đạc và so sánh hiệu suất khi ứng dụng 3 kiến trúc lập trình khác nhau để phân tích đặc tính:
1. Single Process (Đơn tiến trình tuần tự)
2. Multi-threading (Đa luồng trên cùng vùng nhớ)
3. Multi-processing (Đa tiến trình giao tiếp qua Queue)

## Cấu trúc thư mục Script

- **`utils.py`**: Chứa thuật toán Non-Maximum Suppression (NMS) phiên bản NumPy và hệ thống Telemetry Profiler để thống kê FPS, Latency và mức tải CPU theo thời gian thực.
- **`single_inference.py`**: Kiến trúc tiêu chuẩn. Pipeline thực thi tuần tự từ bước Capture I/O -> Preprocess/Inference -> NMS/Display. 
- **`multi_thread_inference.py`**: Kiến trúc Producer-Consumer chia tài nguyên hệ thống thành 2 luồng độc lập: một luồng Camera liên tục đẩy ảnh vào Queue, một luồng AI lấy ảnh ra dự đoán. Giao tiếp qua RAM nội bộ với zero-copy.
- **`multi_process_inference.py`**: Kiến trúc Đa tiến trình, chia 3 khâu Camera, AI và Display ra thành 3 Process hoàn toàn cách ly với nhau về tài nguyên phần cứng (được ép vào các nhân CPU khác nhau thông qua Process Affinity), giao tiếp bằng OS Queue.

## Kết quả thực nghiệm và nhận xét
Ở độ phân giải nhận diện **640x640**:

| Cấu hình | FPS End-to-End | AI Latency (ms) | Hiện tượng nổi bật |
|----------|---------------|----------------|--------------------|
| **Single Process** | 1.85 FPS | ~510 ms | Luồng tuần tự chậm, Camera chờ AI. |
| **Multi-Thread**   | **2.15 FPS** | **~465 ms** | Giải phóng GIL ở TFLite, luồng chạy mượt nhất. |
| **Multi-Process**  | 1.48 - 1.62 FPS | ~598 - 640 ms | Nghẽn I/O (Bottleneck) khi truyền ảnh 640 qua OS Pipe. |

**Nhận xét cốt lõi:**
1. **Multi-Threading (Đa luồng)** mang lại hiệu năng cao nhất trên ảnh phân giải lớn (640x640). Mặc dù Python bị giới hạn bởi GIL, TFLite gọi xuống backend C++ đã tự giải phóng khóa GIL này. Các thread trao đổi hình ảnh với độ trễ gần như bằng 0 (Zero-copy memory).
2. **Multi-Processing (Đa tiến trình)** với độ phân giải lớn bị rơi vào bẫy nghẽn cổ chai IPC (Inter-Process Communication). Mỗi khung hình đẩy từ tiến trình Camera sang tiến trình AI buộc OS phải Serialize và Deserialize (Pickling), khiến Latency tăng vọt.
3. Kỹ thuật đưa Camera vào luồng/tiến trình riêng biệt đi kèm cơ chế tự xả rỗng Queue (Flush) giúp Camera loại bỏ độ trễ và luôn lấy ảnh mới nhất (Zero-Latency I/O).
