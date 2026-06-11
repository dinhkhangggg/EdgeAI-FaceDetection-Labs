import cv2, numpy as np, time, psutil, os, argparse
import tflite_runtime.interpreter as tflite
from utils import Telemetry, nms_numpy

os.environ["QT_QPA_PLATFORM"] = "xcb"

try: psutil.Process().cpu_affinity([0, 1, 2, 3])
except: pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", type=str, default="best_int8.tflite")
    return p.parse_args()

def main():
    args = parse_args()
    
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=4)
    interpreter.allocate_tensors()
    
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    tel = Telemetry(name="Single-Process")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        t_pipeline_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret: break
        h_orig, w_orig = frame.shape[:2]
        t_cap = (time.perf_counter() - t_pipeline_start) * 1000

        t_inf_start = time.perf_counter()
        
        blob = (cv2.resize(frame, (args.imgsz, args.imgsz)).astype(np.float32) / 255.0 * 255 - 128).astype(np.int8)
        
        interpreter.set_tensor(in_idx, np.expand_dims(blob, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(out_idx)[0].transpose()
        
        boxes, scores = [], []
        for row in output:
            score = (row[4] + 128) / 255.0
            
            if score > 0.5:
                cx, cy, w, h = (row[:4] + 128) / 255.0 * args.imgsz
                
                x1 = int((cx - w/2) * w_orig / args.imgsz)
                y1 = int((cy - h/2) * h_orig / args.imgsz)
                
                boxes.append([x1, y1, int(x1 + w*w_orig/args.imgsz), int(y1 + h*h_orig/args.imgsz)])
                scores.append(score)
        
        indices = nms_numpy(boxes, scores)
        t_inf = (time.perf_counter() - t_inf_start) * 1000

        t_disp_start = time.perf_counter()
        
        for i in indices:
            b, s = boxes[i], scores[i]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), tel.color, 2)
            cv2.putText(frame, f"{s:.2f}", (b[0], b[1]-5), 0, 0.6, tel.color, 2)
        
        tel.draw_ui(frame)
        cv2.imshow("Single Inference", frame)
        
        t_now = time.perf_counter()
        tel.update(t_cap, t_inf, (t_now - t_disp_start)*1000, (t_now - t_pipeline_start)*1000)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
