import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ===== CONFIG =====
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "test.jpg"
CONF_THRES = 0.25
IOU_THRES = 0.45

# ===== LOAD MODEL =====
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_dtype = input_details[0]['dtype']
out_dtype = output_details[0]['dtype']

# ===== LOAD IMAGE =====
img = cv2.imread(IMAGE_PATH)
h, w, _ = img.shape

# ===== SCALE AUTO (QUAN TRONG) =====
scale_ui = max(h, w) / 640   # scale theo anh
thickness = int(3 * scale_ui)
font_scale = 1.2 * scale_ui
font_thickness = int(2 * scale_ui)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))

# ===== PREPROCESS =====
if in_dtype == np.int8:
    scale, zp = input_details[0]['quantization']
    input_data = (img_resized / 255.0 / scale + zp).astype(np.int8)
else:
    input_data = (img_resized / 255.0).astype(np.float32)

input_data = np.expand_dims(input_data, axis=0)

# ===== INFERENCE =====
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# ===== DEQUANT =====
if out_dtype == np.int8:
    scale, zp = output_details[0]['quantization']
    output = (output.astype(np.float32) - zp) * scale

# ===== DECODE =====
boxes = []
scores = []

for i in range(output.shape[1]):
    conf = output[4, i]
    if conf < CONF_THRES:
        continue
    
    x, y, bw, bh = output[0:4, i]
    
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    
    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    scores.append(float(conf))

# ===== NMS =====
indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES)

# ===== DRAW =====
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w_box, h_box = boxes[i]
        conf = scores[i]
        
        color = (255, 0, 0)  # xanh duong
        
        # ===== DRAW BOX =====
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), color, thickness)
        
        # ===== LABEL =====
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thickness)
        # padding cho dep
        pad = int(6 * scale_ui)
        
        # ===== NEN =====
        cv2.rectangle(img,
                      (x, y - th - pad),
                      (x + tw + pad, y),
                      color,
                      -1)
                      
        # ===== TEXT =====
        cv2.putText(img,
                    label,
                    (x + int(pad/2), y - int(pad/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA)
                    
    print("Detected:", len(indices), "faces")
else:
    print("Khong detect duoc gi")

# ===== SAVE =====
cv2.imwrite("result.jpg", img)
print("Saved result.jpg")
