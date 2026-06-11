# Lab 5 Practical Report
**Topic: Optimizing Face Detection System (YOLOv8) on Raspberry Pi 4 Using Multi-Process Architecture**

## Lab Objectives
The goal of this lab is to evaluate and solve the "bottleneck" problem of the Single-Process architecture when running AI models on embedded devices (Raspberry Pi 4). 
We restructure the source code into a Multi-Process form with a Queue mechanism to fully utilize the 4 physical cores of the ARM processor, aiming to optimize Throughput and Latency.

## Script Folder Structure

- **`utils.py`**: Contains the `Telemetry` class to measure real-time performance (FPS, Latency) and the `nms_numpy` function (Non-Maximum Suppression rewritten in Numpy to reduce CPU load).
- **`single_inference.py`**: Baseline architecture source code (Sequential Single-Process). 
- **`multi_inference.py`**: Load-balancing Multi-Process architecture source code using Queues (1 Core reading Camera, 2 Cores running inference, 1 Core displaying).
- **`multi_inference_2core.py`**: The most optimized load-sharing source code (Data Parallelism & Round-Robin Load Balancing) on 2 separate AI processes.
- **`benchmark_compare.py`**: Tool to evaluate Pure Inference Latency on static images, measuring P50, P90, P99 to confirm system stability.

## Main Experimental Results

The problem shows a clear trade-off between design structure and performance when running 640x640 vs 320x320 resolutions:

### 1. At 640x640 Resolution
- The Multi-process model yielded a lower FPS (1.48 FPS) than Single-Process (1.85 FPS). 
- Reason: Transferring overly large Tensors between CPU cores causes memory bandwidth "bottlenecks" (IPC Overhead and Pickling/Unpickling).

### 2. At 320x320 Resolution
When the transmission load is reduced, the Multi-process architecture demonstrates its true power:
- **Single 320**: Reaches ~5.12 FPS.
- **Multi-Process 2 Core / 2 Images (320)**: Peaks at ~7.05 - 7.45 FPS. 
- I/O Latency (Capture) is maximally reduced; parallel processes are not blocked by Python's GIL.

### 3. Pure Processing Capacity (Pure Inference Benchmark)
- **Mean Latency (Pure)**: ~281.91 ms (with 640x640 image), equivalent to a theoretical maximum power of ~3.55 FPS per core.
- The system maintains thermal stability without Thermal Throttling, as P50 and P99 differ by < 3.5%.

## Conclusion
- **High Resolution (640)**: Multi-process should not be used as inter-process I/O overhead outweighs CPU load-sharing benefits. Prioritize Single-process.
- **Low Resolution (320)**: Multi-process is mandatory to achieve Real-time speeds. The Hybrid architecture (2 Cores running separate AI processing alternating images) is the optimal architecture to utilize the Raspberry Pi 4.
