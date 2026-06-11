# 🚀 Edge AI & IoT (AIoT) on Raspberry Pi 4
**Repository tổng hợp mã nguồn và báo cáo thực hành (LAB 1 - LAB 7)**

Repository này chứa toàn bộ mã nguồn, tài liệu và các script tối ưu hóa hiệu năng (Benchmarking, Multithreading, Multiprocessing, MQTT) được xây dựng trong quá trình thực hành bộ môn AIoT (Trí tuệ nhân tạo vạn vật) triển khai trên phần cứng Edge Device (Raspberry Pi 4 8GB).

---

## 📂 Cấu trúc Repository

Hệ thống bài thực hành được chia thành 7 LAB, với mức độ phức tạp và tối ưu hóa tăng dần:

### 🔹 [LAB 1: Triển khai Image Classification cơ bản](./LAB1)
- Trích xuất và chạy mô hình mạng neural (MobileNetV2, ResNet) trên thiết bị nhúng.
- Làm quen với TensorFlow Lite (TFLite) và cấu trúc pipeline phân loại hình ảnh cơ bản.

### 🔹 [LAB 2: Khám phá Face Detection truyền thống](./LAB2)
- Cài đặt và đo lường hiệu suất của các thuật toán nhận diện khuôn mặt như Haar Cascade và MTCNN.
- So sánh các phương pháp trích xuất đặc trưng truyền thống với mạng học sâu.

### 🔹 [LAB 3: Triển khai YOLOv8 TFLite](./LAB3)
- Đưa mô hình phát hiện đối tượng/khuôn mặt hiện đại (YOLOv8) lên Raspberry Pi.
- Áp dụng kỹ thuật Lượng tử hóa mô hình (Post-Training Quantization - INT8) để giảm kích thước model và tăng tốc độ suy luận.

### 🔹 [LAB 4: Tối ưu hoá Bộ phân loại và Phân tích Bottleneck](./LAB4)
- Triển khai và đo lường sự khác biệt về End-to-End Latency.
- Đánh giá năng lực của phần cứng khi chịu tải liên tục bằng các script benchmark vòng lặp.

### 🔹 [LAB 5: Multi-Process Pipeline & Queue](./LAB5)
- Đập bỏ kiến trúc Single-Process (Đơn tiến trình) truyền thống.
- Thiết kế hệ thống Đa tiến trình (Multi-Process) sử dụng `multiprocessing.Queue` để phân tải I/O (Camera) và tính toán (AI) lên các nhân CPU (Core) khác nhau của Raspberry Pi.
- Phân tích hiện tượng nghẽn băng thông bộ nhớ (IPC Bottleneck).

### 🔹 [LAB 6: Multi-Threading vs Multi-Processing](./LAB6)
- Đánh giá toàn diện sự khác biệt giữa xử lý Đa luồng (Multi-threading) và Đa tiến trình.
- Kỹ thuật vượt rào GIL (Global Interpreter Lock) của Python sử dụng backend C++ của TFLite, đạt tới kiến trúc Zero-copy memory tối ưu nhất cho thiết bị biên.

### 🔹 [LAB 7: Tích hợp Edge AI và MQTT (AIoT Pipeline)](./LAB7)
- Đưa hệ thống lên mạng lưới IoT bằng giao thức MQTT.
- Xây dựng Flask Dashboard trên Laptop để theo dõi thông số Telemetry (FPS, Latency, số lượng người) theo thời gian thực từ Raspberry Pi truyền về.
- Xử lý các vấn đề suy giảm hiệu suất khi kết hợp Pub/Sub đồng thời với luồng AI.

---

## 🛠 Yêu cầu Hệ thống (Hardware & Software)

- **Phần cứng**: Raspberry Pi 4 (khuyến nghị bản 4GB/8GB RAM) hoặc Raspberry Pi 5.
- **Hệ điều hành**: Raspberry Pi OS (64-bit) / Ubuntu Server.
- **Camera**: USB Webcam hoặc Pi Camera (Giao tiếp qua V4L2).
- **Phần mềm / Thư viện**:
  - `Python >= 3.9`
  - `opencv-python` (Hỗ trợ xử lý ảnh)
  - `tflite_runtime` (Bản rút gọn của TensorFlow, tối ưu cho thiết bị Edge)
  - `paho-mqtt` (Giao tiếp IoT)
  - `Flask` (Xây dựng Dashboard Web)

---

## ⚙️ Hướng dẫn cài đặt môi trường nhanh

Tạo môi trường ảo (Virtual Environment) và cài đặt các gói cần thiết trên Raspberry Pi:

```bash
# 1. Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# 2. Cài đặt các thư viện lõi hệ thống
sudo apt install libgl1-mesa-glx mosquitto mosquitto-clients -y

# 3. Tạo môi trường ảo và kích hoạt
python3 -m venv .venv
source .venv/bin/activate

# 4. Cài đặt Python packages
pip install --upgrade pip
pip install opencv-python numpy paho-mqtt flask
# Lưu ý: Cài đặt tflite-runtime phù hợp với kiến trúc ARM64 của Pi
pip install tflite-runtime
```

---

## 📊 Hệ thống Đo lường và Đánh giá hiệu năng (Telemetry & Benchmark)
Xuyên suốt các Lab (từ Lab 5 đến 7), repository tích hợp một module `utils.py` cực kỳ mạnh mẽ chứa **Bộ giám sát hiệu năng**. Module này chịu trách nhiệm:
- Bóc tách và đo lường độ trễ (Latency) tới mức mili-giây của từng Node: `Capture (I/O)`, `Inference (Compute)`, `Display (Render)`.
- Xác định điểm nghẽn (Bottleneck) chính xác yếu điểm của hệ thống.
- Thực thi thuật toán `NMS` (Non-Maximum Suppression) phiên bản thuần `NumPy` để loại bỏ sự phụ thuộc nặng nề vào TensorFlow backend.

---

*Project được thực hiện và tối ưu dành riêng cho môn học AIoT / Hệ thống nhúng thời gian thực.*