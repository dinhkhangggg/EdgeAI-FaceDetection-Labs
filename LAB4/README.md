# Lab 4 Practical Report
**Topic: Structured Pruning & YOLOv8 Optimization on Raspberry Pi**

## Lab Objectives
This advanced section focuses on the Structured Pruning technique (specifically Channel Pruning) for the YOLOv8 model, to create a truly compact model, thereby optimizing memory and reducing Floating Point Operations (FLOPs) before converting to the quantized TFLite INT8 format for execution on a Raspberry Pi 4.

1. Eliminate 40% of the channels in Convolutional layers.
2. Fine-tune the model after Pruning to restore detection accuracy.
3. Directly compare performance between INT8 and INT8 Pruned at the original 640x640 resolution and a smaller 320x320 resolution.

## Script Folder Structure

- **`01_prune_model.py`**: Source code using PyTorch library to perform direct pruning on the YOLOv8 model's backbone and neck. Reduces the least important 40% of channels.
- **`02_analyze_params_size.py`**: Script comparing total parameters (Params) and storage file size (MB) between the original model (`best.pt`) and the pruned model (`pruned.pt`). 
- **`03_approx_flops.py`**: Script estimating the new GFLOPs of the model because standard thop/ultralytics cannot extract Pruned parameters via direct forward pass.
- **`04_detect_pi_tflite_pruned.py`**: Face Detection pipeline manually written on Pi (handles preprocessing, calling `tflite_runtime`, postprocessing NMS) supporting flexible input image sizes.
- **`05_benchmark_tflite_pruned.py`**: Source code to measure inference time, calculate FPS, and P50, P90, P99 values on the Pi.

## Experimental Summary

| Metric | best.pt | pruned.pt | Improvement (times) |
|---|---|---|---|
| Parameters | 3,011,043 | 1,469,939 | ~2.05 |
| GFLOPs | 8.2 | 4.0 | ~2.05 |
| Size (.pt) | 6.24 MB | 3.15 MB | ~1.98 |

However, when running INT8 vs INT8 Pruned experiments on the **Raspberry Pi** at the same `imgsz=640` resolution:
- **FPS showed no significant change** (still around 1.55 - 1.56 FPS). 
- **Reason:** On an ARM CPU, structured pruning helps reduce parameters and operations, but the system still executes as a Dense Kernel. Reducing channels does not always completely optimize memory access times and the execution pipeline, leading to practical speed changes that do not proportionally match theoretical reductions.

## Highlight: Resizing Input
The largest speed improvement comes from reducing the input resolution `imgsz` from 640 to 320:
- **Speed:** Leaped from 1.55 FPS to **6.3 FPS** (4 times faster). Latency reduced to ~157 ms.
- **Reliability:** Because quantization involves rounding factors, inputting smaller images simplifies features, reducing Quantization noise, which significantly increases Confidence from 0.5 to 0.91 on the INT8 set.
