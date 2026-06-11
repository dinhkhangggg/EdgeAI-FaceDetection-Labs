# Lab 2 Practical Report
**Topic: Training Face Detection Model with YOLOv8**

## Lab Objectives
Train an AI model (YOLOv8) to automatically detect faces from the self-collected dataset in Lab 1.

## Dataset Description
Inherited the dataset from Lab 1 (already labeled and converted to YOLO format). The total number of images is 232.

## Step-by-step Guide

### Step 1: Data Preparation
Run the script `01_prepare_yolo_dataset.py` to split the dataset into Train/Val/Test with corresponding ratios (70/15/15) and automatically generate the `data.yaml` configuration file.
```bash
python 01_prepare_yolo_dataset.py
```
**Splitting Results:**
- Train: ~69.83%
- Val: ~14.65%
- Test: ~15.52%

### Step 2: Training Process
Use the YOLOv8 Nano version (`yolov8n.pt`) because the dataset size is small. This is suitable for future deployment on Edge devices like Raspberry Pi 4 to prevent VRAM overflow.
Command to run training on GPU:
```bash
yolo detect train data=dataset_yolo/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16 device=0
```
*(Hyperparameters: `imgsz=640`, `epochs=50`, `batch=16`)*

### Step 3: Model Evaluation
The results after 50 training epochs achieved very good quality:
- **Precision (P):** 96.9%
- **Recall (R):** 90.3%
- **mAP50:** 90.8%
- **mAP50-95:** 77.6%
The model generalized well and did not suffer from Overfitting. The true face detection rate reached 91% of the actual faces in the test set.

### Step 4: Inference
**Predict on the Test image directory:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset_yolo/images/test conf=0.25 save=True
```

**Real-time prediction via Webcam:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0 conf=0.25
```
*(The estimated processing speed of the model is about 3.9ms/image).*

### Step 5: Deployment on Edge Device (ONNX Export)
To ensure the model operates stably on a Raspberry Pi or platforms with limited libraries, export the weights to the ONNX standard (cross-platform compatibility):
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```
The file `best.onnx` will be generated with a lightweight size (about 11.7 MB).

## Future Improvements
- Perform Data Augmentation, capture photos at obscured angles and complex lighting.
- Use LabelImg to re-verify automatically generated labels.
- Use `yolov8s.pt` combined with a higher number of epochs (e.g., 100) for evaluation.
