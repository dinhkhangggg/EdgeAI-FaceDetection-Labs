# Báo Cáo Thực Hành Lab 7
**Chủ đề: MQTT + EDGE AI DEPLOYMENT TRÊN RASPBERRY PI (Ứng dụng Face Detection)**

## Mục tiêu bài Lab
Triển khai phát hiện khuôn mặt ngay trên Raspberry Pi, sau đó gửi kết quả suy luận theo thời gian thực qua giao thức MQTT để tích hợp với Dashboard Web tĩnh hoặc hệ thống IoT.

## Cấu trúc thư mục Script

- **`face_detection_mqtt.py`**: Ứng dụng chính cho việc đọc camera, phát hiện khuôn mặt với thuật toán Haar Cascade (OpenCV) và đóng gói kết quả, gửi dữ liệu telemetry qua MQTT.
- **`mqtt_subscriber.py`**: Script test cơ bản để chạy trên Terminal/Console lắng nghe kênh MQTT và hiển thị kết quả.
- **`utils_test_camera.py`**: Tiện ích nhỏ để kiểm tra xem Camera có hoạt động hay đang bị ứng dụng khác chiếm dụng không.
- **`benchmark_face_detection.py`**: Kịch bản dùng để đánh giá năng suất (Benchmark FPS, Latency) của Edge Device với các độ phân giải khác nhau.
- **`app_pi.py`**: Server Flask chạy trên mạch Raspberry Pi để stream camera sang Web.
- **`app_laptop.py`**: Dashboard Flask chạy trên thiết bị (như máy tính/laptop) để xem thông số MQTT thời gian thực đồng thời xem luồng Video trực tiếp từ Pi truyền về, cũng như xử lý nhận diện đối với ảnh tĩnh.

## Hướng dẫn thiết lập hệ thống

### Môi trường trên Raspberry Pi
Cài đặt MQTT Broker và các gói Python phụ thuộc:
```bash
sudo apt update && sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
pip3 install opencv-python paho-mqtt numpy flask
```

### Các bước chạy
1. Đảm bảo broker Mosquitto đang chạy.
2. Trên máy Pi (Publisher): Chạy lệnh `python3 app_pi.py` hoặc `python3 face_detection_mqtt.py`.
3. Trên máy Laptop (Subscriber): Chạy lệnh `python3 app_laptop.py`. Mở trình duyệt và truy cập vào IP của Flask Server Laptop `http://localhost:5000` (Thay IP MQTT cấu hình trên máy tính nếu Pi không ở chung máy).

## Đánh giá kết quả Benchmark

1. **Ở phân giải thấp (320x240):** FPS đạt 23.61, Latency 42.35 ms. Hoạt động cực kỳ ổn định, là độ phân giải tối ưu nếu chỉ quan trọng tính thời gian thực mà không cần độ nét cao.
2. **Ở phân giải cơ bản (640x480):** FPS đạt 22.34, Latency 44.75 ms. Đây là mức bão hoà cực kỳ lý tưởng về chi tiết ảnh cũng như hiệu năng.
3. **Ở phân giải cao (1280x720):** FPS rớt thảm hại còn 6.86, Latency 145.69 ms. Nút thắt cổ chai Compute-bound đã xuất hiện do quá trình quét Sliding Window tăng vọt, gây nghẽn phần cứng.
4. **Vấn đề GIL của Python khi tích hợp Pub + Sub:** Do MQTT và OpenCV tranh giành tài nguyên xử lý trong môi trường Python, Context Switching liên tục diễn ra khiến FPS sụt giảm trên dưới 50% (Còn 9.25 FPS). Tương lai đòi hỏi giải pháp Multiprocessing để giải quyết triệt để sự cố suy giảm này.
