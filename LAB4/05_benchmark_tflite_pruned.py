import time
import numpy as np
import cv2
import os

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

# ===== CONFIG =====
MODEL_PATH = "best_float32.tflite" 
IMAGE_PATH = "test.jpg"        
WARMUP = 15      
N_RUNS = 100     

def run_benchmark():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
        print(f"Error: Missing {MODEL_PATH} or {IMAGE_PATH}")
        return
        
    itp = tflite.Interpreter(model_path=MODEL_PATH)
    itp.allocate_tensors()
    
    i_det = itp.get_input_details()
    o_det = itp.get_output_details()
    shape = i_det[0]['shape']
    dtype = i_det[0]['dtype']
    
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[2], shape[1]))
    
    if dtype == np.int8 or dtype == np.uint8:
        scale, zp = i_det[0]['quantization']
        input_data = (img / 255.0 / scale + zp).astype(dtype)
    else:
        input_data = (img / 255.0).astype(np.float32)
        
    input_data = np.expand_dims(input_data, axis=0)
    
    # ===== WARMUP =====
    for _ in range(WARMUP):
        itp.set_tensor(i_det[0]['index'], input_data)
        itp.invoke()
        
    # ===== BENCHMARK =====
    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        itp.set_tensor(i_det[0]['index'], input_data)
        itp.invoke()
        _ = itp.get_tensor(o_det[0]['index'])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    latencies = np.sort(np.array(latencies))
    mean_val = np.mean(latencies)
    fps = 1000.0 / mean_val
    
    # ===== IN KET QUA =====
    print("\n" + "="*45)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 45)
    print(f"{'Model File':<25} | {MODEL_PATH}")
    print(f"{'Mean Latency (ms)':<25} | {mean_val:<15.2f}")
    print(f"{'P50 - Median (ms)':<25} | {np.percentile(latencies, 50):<15.2f}")
    print(f"{'P90 (ms)':<25} | {np.percentile(latencies, 90):<15.2f}")
    print(f"{'P99 (ms)':<25} | {np.percentile(latencies, 99):<15.2f}")
    print(f"{'Approx FPS':<25} | {fps:<15.2f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    run_benchmark()
