import threading, queue, cv2, time, numpy as np, os, argparse
import tflite_runtime.interpreter as tflite 
from utils import Telemetry, nms_numpy
 
os.environ["QT_QPA_PLATFORM"] = "xcb"
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", type=str, default="best_int8.tflite")
    return p.parse_args()
 
args = parse_args()
 
frame_queue = queue.Queue(maxsize=1) 
result_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
 
def capture_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        while not frame_queue.empty():
            try: frame_queue.get_nowait()
            except: break
            
        frame_queue.put((frame, time.perf_counter()))
        time.sleep(0.005)
    cap.release()
 
def inference_thread():
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=3)
    interpreter.allocate_tensors()
    in_idx, out_idx = interpreter.get_input_details()[0]['index'], interpreter.get_output_details()[0]['index']
 
    while not stop_event.is_set():
        try:
            frame, t_cap = frame_queue.get(timeout=1.0)
            h_orig, w_orig = frame.shape[:2]
            t_inf_s = time.perf_counter()
            
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
            
            if result_queue.full():
                try: result_queue.get_nowait()
                except: pass
            result_queue.put((frame, boxes, scores, indices, t_cap, t_inf_s, time.perf_counter()))
        except queue.Empty: continue
 
def main():
    tel = Telemetry(name="Multi-Threading")
    
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t1.start(); t2.start()
 
    try:
        while True:
            try:
                frame, boxes, scores, indices, t_cap, t_inf_s, t_inf_e = result_queue.get(timeout=0.1)
            except queue.Empty: continue
 
            t_disp_s = time.perf_counter()
            for i in indices:
                b, s = boxes[i], scores[i]
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), tel.color, 2)
                cv2.putText(frame, f"{s:.2f}", (b[0], b[1]-5), 0, 0.6, tel.color, 2)
            
            tel.draw_ui(frame)
            cv2.imshow("Thread Inference", frame)
            
            t_now = time.perf_counter()
            tel.update((t_inf_s - t_cap)*1000, (t_inf_e - t_inf_s)*1000, 
                       (t_now - t_disp_s)*1000, (t_now - t_cap)*1000)
 
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
