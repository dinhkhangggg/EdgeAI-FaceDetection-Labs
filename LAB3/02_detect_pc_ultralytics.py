from ultralytics import YOLO

MODEL_TFLITE = "best_int8.tflite"   # change to the correct filename
SOURCE = "test_images"             # can be 'test.jpg' or folder

# Ultralytics handles pre-processing + post-processing (NMS) for TFLite
model = YOLO(MODEL_TFLITE)
results = model.predict(source=SOURCE, imgsz=640, conf=0.25, save=True)
print("Done. Output in runs/detect/predict/")
