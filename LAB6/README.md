# Lab 6 Practical Report
**Topic: Optimizing Face Detection with Multithreading and Multiprocessing on Raspberry Pi**

## Lab Objectives
Deploy Face Detection using YOLOv8 with the TFLite INT8 model on a Raspberry Pi board. Simultaneously measure and compare performance when applying 3 different programming architectures to analyze their characteristics:
1. Single Process (Sequential execution)
2. Multi-threading (Shared memory space)
3. Multi-processing (IPC via Queue)

## Script Folder Structure

- **`utils.py`**: Contains the NumPy version of the Non-Maximum Suppression (NMS) algorithm and a Telemetry Profiler system to track FPS, Latency, and CPU load in real-time.
- **`single_inference.py`**: Baseline architecture. Pipeline executes sequentially from Capture I/O -> Preprocess/Inference -> NMS/Display. 
- **`multi_thread_inference.py`**: Producer-Consumer architecture dividing system resources into 2 independent threads: one Camera thread continuously pushing images to a Queue, and an AI thread popping images for prediction. Communication via internal RAM with zero-copy.
- **`multi_process_inference.py`**: Multi-process architecture, splitting the 3 stages of Camera, AI, and Display into 3 completely isolated hardware resource Processes (pinned to different CPU cores via Process Affinity), communicating via OS Queue.

## Experimental Results and Remarks
At **640x640** detection resolution:

| Configuration | End-to-End FPS | AI Latency (ms) | Notable Phenomenon |
|----------|---------------|----------------|--------------------|
| **Single Process** | 1.85 FPS | ~510 ms | Slow sequential thread, Camera waiting for AI. |
| **Multi-Thread**   | **2.15 FPS** | **~465 ms** | GIL released in TFLite, smoothest running thread. |
| **Multi-Process**  | 1.48 - 1.62 FPS | ~598 - 640 ms | I/O Bottleneck when passing 640 image via OS Pipe. |

**Core Remarks:**
1. **Multi-Threading** brings the highest performance on high-resolution images (640x640). Although Python is restricted by the GIL, TFLite calls the C++ backend which automatically releases this GIL lock. Threads exchange images with near-zero latency (Zero-copy memory).
2. **Multi-Processing** with large resolutions falls into the IPC (Inter-Process Communication) bottleneck trap. Every frame pushed from the Camera process to the AI process forces the OS to Serialize and Deserialize (Pickling), causing Latency to spike.
3. The technique of isolating the Camera into a separate thread/process coupled with an auto-flush mechanism helps the Camera eliminate latency and always fetch the newest image (Zero-Latency I/O).
