# Lab 3 Practical Report
**Topic: INT8 Quantization and YOLOv8 Face Detection on Raspberry Pi**

## Lab Objectives
1. Deploy a Face Detection model based on the YOLO architecture on the Raspberry Pi platform.
2. Optimize the model using Quantization (INT8) techniques to reduce size and increase processing speed on embedded hardware.
3. Evaluate real-world performance through metrics: Latency and FPS (Frames Per Second) to compare the Float32 and INT8 versions.

## Environment Setup & Model Export
- Install `ultralytics`, `ai-edge-litert` (or `tflite-runtime`), `opencv-python`.
- To convert the model (PTQ - Post Training Quantization) to INT8 on a PC:
  ```bash
  yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True data=dataset_yolo/data.yaml
  ```
- The received results include 2 main files for testing: `best_float32.tflite` and `best_full_integer_quant.tflite` (or `best_int8.tflite`).

## Script Guide

This folder contains 4 scripts extracted from the report:

1. **`01_check_model_int8.py`**: 
   Used to quickly inspect the TFLite file. Provides detailed reports on Input/Output tensor types and quantization parameters (scale, zero-point). 

2. **`02_detect_pc_ultralytics.py`**: 
   Runs predictions on a PC using the `ultralytics` library directly. The pre/post-processing pipeline is handled automatically.

3. **`03_detect_pi_tflite.py`**: 
   A dedicated script to run on Raspberry Pi using the `tflite_runtime` (or `ai_edge_litert`) library. The detection pipeline must be written manually: size normalization, input quantization (if INT8), calling Interpreter, dequantization, Bounding Box recalculation, and applying NMS (Non-Maximum Suppression). 

4. **`04_benchmark_tflite.py`**: 
   Used to benchmark model speed (FPS, Latency, P50, P90, P99) of the TFLite model (Float32 or INT8) across 100 iterations. Can be run on PC or Pi for comparison. 

## Summary of Experimental Results

1. **Model Size:**
   - INT8 is about **3.7 times** smaller than Float32 (due to converting from 32-bit to 8-bit).
2. **On PC (x86_64):**
   - **Speed:** INT8 reached ~39 FPS (25.6 ms), Float32 reached ~13 FPS (75 ms) -> Nearly **3 times faster**. (PC has excellent support for SIMD/AVX instruction sets for integers).
   - **Accuracy:** Because this is Post-Training Quantization (PTQ) without QAT applied, the Confidence of INT8 decreased significantly (from 0.95 to about 0.50).
3. **On Raspberry Pi (ARM):**
   - **Speed:** INT8 reached ~1.55 FPS (645 ms), Float32 reached ~1.11 FPS (902 ms) -> Speed only increased by **1.4 times**. The reason is that ARM architectures have more limited SIMD support, and the CPU has to carry additional overhead from Python and image pre/post-processing.
   - **Accuracy:** Similar to the PC, the quantized version experienced a drop in confidence. However, Float32 on Pi (0.93) is also slightly off compared to PC (0.95) because the manually written pre/post-processing pipeline via OpenCV is not 100% identical to the hidden functions inside the Ultralytics framework.
