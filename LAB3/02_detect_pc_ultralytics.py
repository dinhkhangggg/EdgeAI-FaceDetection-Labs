from ultralytics import YOLO

MODEL_TFLITE = "best_int8.tflite"   # đổi đúng tên file
SOURCE = "test_images"             # có thể là 'test.jpg' hoặc folder

# Ultralytics tự xử lý tiền xử lý + hậu xử lý (NMS) cho TFLite
model = YOLO(MODEL_TFLITE)
results = model.predict(source=SOURCE, imgsz=640, conf=0.25, save=True)
print("Done. Output in runs/detect/predict/")
