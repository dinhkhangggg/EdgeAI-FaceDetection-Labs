import json 
import time 
import cv2 
import paho.mqtt.client as mqtt 

# MQTT Configuration
BROKER = "localhost" 
PORT = 1883 
TOPIC = "edgeai/face_detection"

# Camera and Logic Configuration
CAMERA_INDEX = 0 
FRAME_WIDTH = 640 
FRAME_HEIGHT = 480 
PUBLISH_INTERVAL = 1.0 
CONFIRM_FRAMES = 3 

# Setup MQTT Connection
client = mqtt.Client() 
client.connect(BROKER, PORT, 60) 

# Load Haar Cascade filter
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Camera Configuration
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Please check the camera and CAMERA_INDEX.") 

last_publish = 0.0 
detection_streak = 0 
frame_count = 0 
t0 = time.time() 

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera.")
            time.sleep(0.1)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40) 
        )
        
        face_count = len(faces)
        if face_count > 0:
            detection_streak += 1 
        else:
            detection_streak = 0 
            
        has_face = 1 if detection_streak >= CONFIRM_FRAMES else 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        frame_count += 1 
        elapsed = max(time.time() - t0, 1e-6) 
        
        fps = frame_count / elapsed 
        
        cv2.putText(frame, f"Faces: {face_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"MQTT status: {has_face}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                    
        now = time.time() 
        if now - last_publish >= PUBLISH_INTERVAL: 
            payload = {
                "has_face": has_face,
                "face_count": int(face_count), 
                "fps": round(fps, 2), 
                "timestamp": now 
            }
            client.publish(TOPIC, json.dumps(payload)) 
            print("Publish:", payload)
            last_publish = now
            
        cv2.imshow("Face Detection Edge AI + MQTT", frame)
        key = cv2.waitKey(1) & 0xFF 
        if key == ord("q"): 
            break
finally:
    cap.release() 
    cv2.destroyAllWindows() 
    client.disconnect()
