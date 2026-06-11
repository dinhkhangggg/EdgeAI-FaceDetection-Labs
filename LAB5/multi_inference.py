import multiprocessing as mp
import queue
import cv2, numpy as np, time, psutil, os, argparse
import tflite_runtime.interpreter as tflite
from utils import Telemetry, nms_numpy

os.environ["QT_QPA_PLATFORM"] = "xcb"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--queue-size", type=int, default=2) 
    p.add_argument("--model", type=str, default="best_int8.tflite")
    return p.parse_args()

def capture_worker(q_in, stop_event):
    try: psutil.Process().cpu_affinity([1])
    except: pass
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                while not q_in.empty():
                    q_in.get_nowait()
            except queue.Empty:
                pass
            
            try:
                q_in.put((frame, time.perf_counter()), block=False)
            except queue.Full:
                pass
            
            time.sleep(0.01)
    cap.release()

def inference_worker(model_path, imgsz, q_in, q_out, stop_event):
    try: psutil.Process().cpu_affinity([2, 3])
    except: pass
    
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    in_idx, out_idx = interpreter.get_input_details()[0]['index'], interpreter.get_output_details()[0]['index']

    while not stop_event.is_set():
        try:
            frame, t_cap = q_in.get(timeout=0.5)
        except queue.Empty:
            continue
            
        h_orig, w_orig = frame.shape[:2]
        t_inf_s = time.perf_counter()
        
        blob = (cv2.resize(frame, (imgsz, imgsz)).astype(np.float32) / 255.0 * 255 - 128).astype(np.int8)
        interpreter.set_tensor(in_idx, np.expand_dims(blob, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(out_idx)[0].transpose()
        
        boxes, scores = [], []
        for row in output:
            score = (row[4] + 128) / 255.0
            if score > 0.5:
                cx, cy, w, h = (row[:4] + 128) / 255.0 * imgsz
                x1 = int((cx - w/2) * w_orig / imgsz)
                y1 = int((cy - h/2) * h_orig / imgsz)
                boxes.append([x1, y1, int(x1 + w*w_orig/imgsz), int(y1 + h*h_orig/imgsz)])
                scores.append(score)
        
        indices = nms_numpy(boxes, scores)
        t_inf_e = time.perf_counter()

        try:
            if q_out.full():
                q_out.get_nowait()
        except queue.Empty:
            pass
            
        try:
            q_out.put((frame, boxes, scores, indices, t_cap, t_inf_s, t_inf_e), block=False)
        except queue.Full:
            pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    
    try: psutil.Process().cpu_affinity([0])
    except: pass
    
    q_in = mp.Queue(maxsize=args.queue_size)
    q_out = mp.Queue(maxsize=2)
    stop_event = mp.Event()
    tel = Telemetry(name=f"MP-Queue (Sz={args.queue_size})")

    p_cap = mp.Process(target=capture_worker, args=(q_in, stop_event), daemon=True)
    p_inf = mp.Process(target=inference_worker, args=(args.model, args.imgsz, q_in, q_out, stop_event), daemon=True)
    p_cap.start(); p_inf.start()

    try:
        while True:
            try:
                frame, boxes, scores, indices, t_cap, t_inf_s, t_inf_e = q_out.get(timeout=0.1)
            except queue.Empty:
                continue

            t_disp_s = time.perf_counter()
            
            for i in indices:
                b, s = boxes[i], scores[i]
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), tel.color, 2)
                cv2.putText(frame, f"{s:.2f}", (b[0], b[1]-5), 0, 0.6, tel.color, 2)

            tel.draw_ui(frame)
            cv2.imshow("MP Queue Inference", frame)
            
            t_now = time.perf_counter()
            tel.update((t_inf_s - t_cap)*1000, (t_inf_e - t_inf_s)*1000, 
                       (t_now - t_disp_s)*1000, (t_now - t_cap)*1000)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        stop_event.set()
        p_cap.terminate(); p_inf.terminate()
        cv2.destroyAllWindows()
