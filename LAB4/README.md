# Báo Cáo Thực Hành Lab 4
**Chủ đề: Structured Pruning & Tối ưu hóa YOLOv8 trên Raspberry Pi**

## Mục tiêu bài Lab
Phần nâng cao này tập trung vào kỹ thuật Structured Pruning (cụ thể là Channel Pruning) cho mô hình YOLOv8, nhằm tạo ra một mô hình nhỏ gọn thực sự, qua đó tối ưu bộ nhớ và giảm số lượng phép toán (FLOPs) trước khi chuyển đổi sang định dạng lượng tử hóa TFLite INT8 để chạy trên thiết bị Raspberry Pi 4.

1. Thực hiện loại bỏ bớt 40% số kênh (channels) ở các lớp Convolution.
2. Fine-tune mô hình sau Pruning để phục hồi độ chính xác nhận diện.
3. So sánh trực tiếp hiệu năng giữa INT8, INT8 Pruned ở độ phân giải gốc 640x640 và phân giải nhỏ 320x320.

## Cấu trúc thư mục Script

- **`01_prune_model.py`**: Mã nguồn dùng thư viện PyTorch để thực hiện prune trực tiếp trên mạng backbone và neck của mô hình YOLOv8. Giảm tải 40% channel ít quan trọng nhất.
- **`02_analyze_params_size.py`**: Script so sánh tổng số tham số (Params) và dung lượng tệp lưu trữ (MB) giữa mô hình gốc (`best.pt`) và mô hình cắt tỉa (`pruned.pt`). 
- **`03_approx_flops.py`**: Script ước lượng GFLOPs mới của mô hình do thop/ultralytics thông thường không lấy được thông số Pruned qua forward pass trực tiếp.
- **`04_detect_pi_tflite_pruned.py`**: Pipeline nhận diện Face Detection viết thủ công trên Pi (xử lý tiền xử lý, gọi `tflite_runtime`, hậu xử lý NMS) hỗ trợ cả kích thước ảnh đầu vào linh hoạt.
- **`05_benchmark_tflite_pruned.py`**: Mã nguồn đo đạc thời gian suy luận, tính FPS, và các giá trị P50, P90, P99 trên Pi.

## Tổng kết thực nghiệm

| Thông số | best.pt | pruned.pt | Cải thiện (lần) |
|---|---|---|---|
| Số tham số | 3,011,043 | 1,469,939 | ~2.05 |
| GFLOPs | 8.2 | 4.0 | ~2.05 |
| Dung lượng (.pt) | 6.24 MB | 3.15 MB | ~1.98 |

Tuy nhiên, khi chạy thực nghiệm INT8 vs INT8 Pruned trên **Raspberry Pi** ở cùng độ phân giải `imgsz=640`:
- **FPS không có sự thay đổi rõ rệt** (vẫn quanh mức 1.55 - 1.56 FPS). 
- **Nguyên nhân:** Trên CPU ARM, structured pruning giúp giảm tham số và phép toán nhưng hệ thống vẫn thực thi dưới dạng Dense Kernel. Việc giảm số kênh đôi khi không tối ưu được hoàn toàn thời gian truy cập bộ nhớ và pipeline thực thi, dẫn đến tốc độ trên thiết bị thực tế thay đổi không tương xứng với lý thuyết cắt giảm.

## Điểm nhấn: Resize Input
Việc cải thiện tốc độ lớn nhất đến từ việc giảm độ phân giải đầu vào `imgsz` từ 640 xuống 320:
- **Tốc độ:** Nhảy vọt từ 1.55 FPS lên **6.3 FPS** (Nhanh hơn gấp 4 lần). Latency giảm còn ~157 ms.
- **Độ tin cậy:** Do lượng tử hoá có yếu tố làm tròn, việc đưa ảnh đầu vào nhỏ hơn lại khiến đặc trưng đơn giản hóa, giảm lỗi lượng tử hóa (Quantization noise), giúp Confidence tăng đáng kể từ 0.5 lên 0.91 trên tập INT8.
