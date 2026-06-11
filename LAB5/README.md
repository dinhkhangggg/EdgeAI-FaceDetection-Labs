# Báo Cáo Thực Hành Lab 5
**Chủ đề: Tối ưu hóa hệ thống nhận diện khuôn mặt (YOLOv8) trên Raspberry Pi 4 bằng kiến trúc Multi-Process**

## Mục tiêu bài Lab
Mục tiêu của bài thực hành này là đánh giá và giải quyết vấn đề "nút thắt cổ chai" (bottleneck) của kiến trúc Single-Process khi chạy mô hình AI trên thiết bị nhúng (Raspberry Pi 4). 
Chúng ta tái cấu trúc mã nguồn sang dạng Đa tiến trình (Multi-Process) với cơ chế Hàng đợi (Queue) để tận dụng tối đa 4 nhân vật lý của vi xử lý ARM, nhằm tối ưu hóa thông lượng (Throughput) và độ trễ (Latency).

## Cấu trúc thư mục Script

- **`utils.py`**: Chứa lớp `Telemetry` để đo lường hiệu năng thời gian thực (FPS, Latency) và hàm `nms_numpy` (Non-Maximum Suppression viết lại bằng Numpy để giảm tải CPU).
- **`single_inference.py`**: Mã nguồn kiến trúc Baseline (Đơn tiến trình tuần tự). 
- **`multi_inference.py`**: Mã nguồn kiến trúc Multi-Process phân tải sử dụng Queue (1 Core đọc Camera, 2 Core chạy suy luận, 1 Core hiển thị).
- **`multi_inference_2core.py`**: Mã nguồn tối ưu nhất chia tải (Data Parallelism & Round-Robin Load Balancing) trên 2 tiến trình AI riêng biệt.
- **`benchmark_compare.py`**: Công cụ đánh giá độ trễ xử lý suy luận thuần túy (Pure Inference Latency) trên ảnh tĩnh, đo đạc P50, P90, P99 để xác nhận tính ổn định hệ thống.

## Kết quả thực nghiệm chính

Bài toán cho thấy sự đánh đổi rõ rệt giữa cấu trúc thiết kế và hiệu suất khi chạy ảnh độ phân giải 640x640 và 320x320:

### 1. Ở độ phân giải 640x640
- Mô hình Đa tiến trình (Multi-process) lại cho FPS (1.48 FPS) thấp hơn Single-Process (1.85 FPS). 
- Nguyên nhân: Việc truyền tải Tensor quá lớn giữa các nhân CPU gây ra "nút thắt cổ chai" băng thông bộ nhớ (IPC Overhead và Pickling/Unpickling).

### 2. Ở độ phân giải 320x320
Khi giảm dung lượng truyền tải, kiến trúc Multi-process thể hiện sức mạnh thực sự:
- **Single 320**: Đạt ~5.12 FPS.
- **Multi-Process 2 Core / 2 Images (320)**: Đạt đỉnh ~7.05 - 7.45 FPS. 
- Độ trễ I/O (Capture) được giảm tối đa, các tiến trình chạy song song không bị Block bởi GIL của Python.

### 3. Năng lực xử lý thuần túy (Pure Inference Benchmark)
- **Mean Latency (Thuần túy)**: ~281.91 ms (với ảnh 640x640), tương đương sức mạnh cực đại lý thuyết của 1 nhân là ~3.55 FPS.
- Hệ thống duy trì được tính ổn định nhiệt và không xảy ra hiện tượng Thermal Throttling khi P50 và P99 chỉ lệch nhau < 3.5%.

## Kết luận
- **Phân giải cao (640)**: Không nên dùng Multi-process vì chi phí I/O liên tiến trình cao hơn lợi ích chia tải CPU. Ưu tiên Single-process.
- **Phân giải thấp (320)**: Bắt buộc dùng Multi-process để đạt tốc độ thời gian thực (Real-time). Kiến trúc lai Hybrid (2 Core chạy AI riêng biệt chia ảnh luân phiên) là kiến trúc tối ưu nhất để tận dụng Raspberry Pi 4.
