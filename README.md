# 🚀 Edge AI & IoT (AIoT) on Raspberry Pi 4
**Comprehensive Repository of Source Code and Lab Reports (LAB 1 - LAB 7)**

This repository contains all source code, documentation, and performance optimization scripts (Benchmarking, Multithreading, Multiprocessing, MQTT) developed during the AIoT (Artificial Intelligence of Things) course, deployed on an Edge Device (Raspberry Pi 4 Model B 8GB).

---

## 📂 Repository Structure

The practical exercises are divided into 7 LABs, with increasing complexity and optimization levels:

### 🔹 [LAB 1: Basic Image Classification Deployment](./LAB1)
- Extract and run neural network models (MobileNetV2, ResNet) on an embedded device.
- Get familiar with TensorFlow Lite (TFLite) and the basic image classification pipeline structure.

### 🔹 [LAB 2: Exploring Traditional Face Detection](./LAB2)
- Install and measure the performance of traditional face detection algorithms like Haar Cascade and MTCNN.
- Compare traditional feature extraction methods with deep learning networks.

### 🔹 [LAB 3: YOLOv8 TFLite Deployment](./LAB3)
- Deploy modern object/face detection models (YOLOv8) on Raspberry Pi.
- Apply Post-Training Quantization (INT8) techniques to reduce model size and accelerate inference speed.

### 🔹 [LAB 4: Classifier Optimization & Bottleneck Analysis](./LAB4)
- Deploy and measure differences in End-to-End Latency.
- Evaluate hardware capacity under continuous load using looped benchmark scripts.

### 🔹 [LAB 5: Multi-Process Pipeline & Queue](./LAB5)
- Dismantle the traditional Single-Process architecture.
- Design a Multi-Process system using `multiprocessing.Queue` to distribute I/O (Camera) and computation (AI) across different CPU cores of the Raspberry Pi.
- Analyze memory bandwidth bottlenecks (IPC Bottleneck).

### 🔹 [LAB 6: Multi-Threading vs Multi-Processing](./LAB6)
- Comprehensively evaluate the differences between Multi-threading and Multi-processing.
- Bypass Python's GIL (Global Interpreter Lock) using the TFLite C++ backend, achieving the optimal Zero-copy memory architecture for edge devices.

### 🔹 [LAB 7: Edge AI and MQTT Integration (AIoT Pipeline)](./LAB7)
- Bring the system to the IoT network via the MQTT protocol.
- Build a Flask Dashboard on a Laptop to monitor real-time Telemetry parameters (FPS, Latency, Face Count) transmitted from the Raspberry Pi.
- Resolve performance degradation issues when combining Pub/Sub concurrently with the AI thread.

---

## 🛠 System Requirements (Hardware & Software)

- **Hardware**: Raspberry Pi 4 (4GB/8GB RAM recommended) or Raspberry Pi 5.
- **OS**: Raspberry Pi OS (64-bit) / Ubuntu Server.
- **Camera**: USB Webcam or Pi Camera (via V4L2).
- **Software / Libraries**:
  - `Python >= 3.9`
  - `opencv-python` (Image processing support)
  - `tflite_runtime` (Lightweight TensorFlow, optimized for Edge devices)
  - `paho-mqtt` (IoT Communication)
  - `Flask` (Web Dashboard builder)

---

## ⚙️ Quick Environment Setup Guide

Create a Virtual Environment and install the required packages on your Raspberry Pi:

```bash
# 1. Update the system
sudo apt update && sudo apt upgrade -y

# 2. Install core system libraries
sudo apt install libgl1-mesa-glx mosquitto mosquitto-clients -y

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Python packages
pip install --upgrade pip
pip install opencv-python numpy paho-mqtt flask
# Note: Install tflite-runtime compatible with the Pi's ARM64 architecture
pip install tflite-runtime
```

---

## 📊 Telemetry & Benchmark System
Throughout the Labs (from Lab 5 to 7), the repository integrates an extremely powerful `utils.py` module containing a **Performance Monitor (Telemetry Profiler)**. This module is responsible for:
- Extracting and measuring millisecond-level Latency for each Node: `Capture (I/O)`, `Inference (Compute)`, `Display (Render)`.
- Identifying bottlenecks to pinpoint exact system weaknesses.
- Executing the `NMS` (Non-Maximum Suppression) algorithm in pure `NumPy` to eliminate heavy reliance on the TensorFlow backend.

---

*This project is implemented and optimized specifically for the AIoT / Real-Time Embedded Systems course.*
