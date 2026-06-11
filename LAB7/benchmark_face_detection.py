import time
import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Khong mo duoc camera.")

num_frames = 200
processed = 0
t0 = time.time()

while processed < num_frames:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _ = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    processed += 1

elapsed = time.time() - t0
fps = processed / elapsed
latency_ms = (elapsed / processed) * 1000
cap.release()

print(f"So frame: {processed}")
print(f"Thoi gian tong: {elapsed:.3f} s")
print(f"FPS trung binh: {fps:.2f}")
print(f"Latency trung binh: {latency_ms:.2f} ms/frame")
