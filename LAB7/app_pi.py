from flask import Flask, Response
import cv2
import time
import json
import paho.mqtt.client as mqtt

app = Flask(__name__)
# ===== MQTT =====
BROKER = "localhost"
PORT = 1883
TOPIC = "edgeai/face_detection"
client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# ===== CAMERA =====
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# ===== MODEL =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===== FPS =====
frame_count = 0
start_time = time.time()
last_publish = 0

def gen_frames():
    global frame_count, start_time, last_publish
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        face_count = len(faces)
        # draw bbox
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # MQTT
        now = time.time()
        if now - last_publish >= 1.0:
            payload = {
                "face_count": face_count,
                "fps": round(fps, 2),
                "timestamp": now
            }
            client.publish(TOPIC, json.dumps(payload))
            last_publish = now
            
        # encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <h1>Pi Camera Stream</h1>
    <img src="/video_feed">
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
