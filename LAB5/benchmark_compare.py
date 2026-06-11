import argparse
import csv
import statistics
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="best_int8.tflite", help="Đường dẫn file model")
    p.add_argument("--source", type=str, default="test.jpg", help="Đường dẫn ảnh tĩnh để test")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước input")
    p.add_argument("--warmup", type=int, default=10, help="Số lần chạy khởi động")
    p.add_argument("--runs", type=int, default=100, help="Số lần chạy tính điểm")
    p.add_argument("--csv", type=str, default="benchmark_summary.csv")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.source):
        print(f"Lỗi: Không tìm thấy ảnh {args.source}. Hãy để 1 tấm ảnh tên test.jpg vào thư mục.")
        return

    interpreter = tflite.Interpreter(model_path=args.model, num_threads=4)
    interpreter.allocate_tensors()
    
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    img = cv2.imread(args.source)
    blob = (cv2.resize(img, (args.imgsz, args.imgsz)).astype(np.float32) / 255.0 * 255 - 128).astype(np.int8)
    input_data = np.expand_dims(blob, axis=0)

    print(f"--- Đang bắt đầu Benchmark mô hình {args.model} ---")
    print(f"Đang chạy khởi động {args.warmup} lần...")
    for _ in range(args.warmup):
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)

    print(f"Đang đo lường {args.runs} lần...")
    latencies = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    mean_ms = statistics.mean(latencies)
    
    p50 = latencies[int(0.50 * (len(latencies) - 1))]
    p90 = latencies[int(0.90 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]
    
    approx_fps = 1000.0 / mean_ms

    print("\n" + "="*30)
    print(f"KẾT QUẢ BENCHMARK ({args.model})")
    print("-" * 30)
    print(f"Mean (Trung bình): {mean_ms:.2f} ms")
    print(f"P50 (Median):      {p50:.2f} ms")
    print(f"P90:               {p90:.2f} ms")
    print(f"P99:               {p99:.2f} ms")
    print(f"FPS dự kiến:       {approx_fps:.2f}")
    print("="*30)

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean(ms)", "P50(ms)", "P90(ms)", "P99(ms)", "FPS"])
        writer.writerow([args.model, f"{mean_ms:.2f}", f"{p50:.2f}", f"{p90:.2f}", f"{p99:.2f}", f"{approx_fps:.2f}"])
    
    print(f"Đã lưu kết quả vào file: {args.csv}")

if __name__ == "__main__":
    main()
