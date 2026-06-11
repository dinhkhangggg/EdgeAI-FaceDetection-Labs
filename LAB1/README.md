# Lab 1 Practical Report
**Topic: Data Collection and Processing for Face Detection**

## Lab Objectives
1. Build a dataset of face images with bounding box labels to train an AI model for Face Detection.
2. Collect real-world data with high diversity in lighting, angles, and subjects.
3. Use the OpenCV library and Haar Cascade algorithm to automate the labeling process.
4. Understand standard dataset storage structures.

## Dataset Folder Structure
The dataset must follow this structure for scripts to process it correctly:
```
face_detection_dataset/
├── images/     # Input images (.jpg, .png...)
├── labels/     # Automatically generated label files (.txt)
└── logs/       # Logs warning about images with no detected faces
```
*Note: Input images should be named following the rule `PersonName_Sequence.jpg` (e.g., `Khang_01.jpg`)*

## Script Usage Guide

This folder contains 2 main scripts extracted from the report:

### 1. Basic Script (`02_auto_label_faces.py`)
Uses the default Haar Cascade model `haarcascade_frontalface_default.xml` to detect frontal faces.
- **How to run:**
  ```bash
  python 02_auto_label_faces.py
  ```
- **Limitations:** Only works well for straight face angles and even lighting. Ineffective for side angles or overexposed images.

### 2. Advanced Script (`02_auto_label_faces_advanced.py`)
This improved version overcomes the basic version's weaknesses using a "3-step strategy":
1. **Step 1:** Frontal view scan.
2. **Step 2:** If not found, profile view scan using `haarcascade_profileface.xml`.
3. **Step 3:** If still not found, flip the image horizontally and scan the profile again to catch opposite side angles.

- **How to run:**
  ```bash
  python 02_auto_label_faces_advanced.py
  ```

## Environment Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- Environment should be run on a Raspberry Pi 4 8GB or a personal computer.

## Conclusions & Recommendations (From the report)
- Using a smartphone for data collection yields better quality than a laptop webcam.
- To fully overcome the limitations of side angles or occlusions, Deep Learning models like MTCNN or YOLO should be considered instead.
