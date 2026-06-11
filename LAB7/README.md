# Lab 7 Practical Report
**Topic: MQTT + EDGE AI DEPLOYMENT ON RASPBERRY PI (Face Detection Application)**

## Lab Objectives
Deploy face detection natively on the Raspberry Pi, then transmit real-time inference results via the MQTT protocol to integrate with a static Web Dashboard or an IoT system.

## Script Folder Structure

- **`face_detection_mqtt.py`**: Main application for reading the camera, detecting faces with the Haar Cascade algorithm (OpenCV), packaging the results, and sending telemetry data via MQTT.
- **`mqtt_subscriber.py`**: Basic test script to run on Terminal/Console listening to the MQTT channel and displaying results.
- **`utils_test_camera.py`**: Small utility to check if the Camera is working or currently occupied by another application.
- **`benchmark_face_detection.py`**: Script used to evaluate the productivity (Benchmark FPS, Latency) of the Edge Device at different resolutions.
- **`app_pi.py`**: Flask server running on the Raspberry Pi board to stream the camera feed to the Web.
- **`app_laptop.py`**: Flask Dashboard running on a remote device (like a laptop/PC) to view real-time MQTT metrics while viewing the Live Video stream sent from the Pi, as well as handling detection processing for static images.

## System Setup Guide

### Environment on Raspberry Pi
Install the MQTT Broker and Python dependencies:
```bash
sudo apt update && sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
pip3 install opencv-python paho-mqtt numpy flask
```

### Execution Steps
1. Ensure the Mosquitto broker is running.
2. On the Pi (Publisher): Run the command `python3 app_pi.py` or `python3 face_detection_mqtt.py`.
3. On the Laptop (Subscriber): Run the command `python3 app_laptop.py`. Open a web browser and access the Flask Server Laptop's IP `http://localhost:5000` (Change the MQTT IP configured on the computer if the Pi is not on the same machine).

## Benchmark Results Evaluation

1. **At low resolution (320x240):** FPS reached 23.61, Latency 42.35 ms. Operations were extremely stable, proving this to be the optimal resolution if only real-time functionality matters over high definition.
2. **At basic resolution (640x480):** FPS reached 22.34, Latency 44.75 ms. This is an highly ideal saturation point for both image detail and performance.
3. **At high resolution (1280x720):** FPS plummeted to 6.86, Latency 145.69 ms. A Compute-bound bottleneck emerged due to a sharp increase in Sliding Window scans, causing hardware throttling.
4. **Python's GIL issue during Pub + Sub integration:** Because MQTT and OpenCV compete for processing resources in the Python environment, continuous Context Switching caused FPS drops of up to 50% (down to 9.25 FPS). Future iterations will require Multiprocessing solutions to permanently resolve this performance degradation issue.
