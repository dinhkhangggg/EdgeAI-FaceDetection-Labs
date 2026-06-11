import multiprocessing as mp
import queue, cv2, numpy as np, time, psutil, os, argparse
import tflite_runtime.interpreter as tflite
from utils import Telemetry, nms_numpy

os.environ["QT_QPA_PLATFORM"] = "xcb"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="best_int8_320.tflite", help="Đường dẫn file TFLite")
    p.add_argument("--queue-size", type=int, default=2, help="Kích thước hàng đợi")
    p.add_argument("--imgsz", type=int, default=320, help="Kích thước ảnh cho AI")
    return p.parse_args()

def capture_worker(q_in_1, q_in_2, stop_event):
    try: psutil.Process().cpu_affinity([0]) 
    except: pass
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    frame_id = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_small = cv2.resize(frame, (640, 480))
            payload = (frame_small, frame_id, time.perf_counter())
            
            try:
                if frame_id % 2 == 0:
                    while not q_in_1.empty(): q_in_1.get_nowait() 
                    q_in_1.put(payload, block=False)
                else:
                    while not q_in_2.empty(): q_in_2.get_nowait()
                    q_in_2.put(payload, block=False)
                    
                frame_id += 1
            except queue.Empty: pass
            except queue.Full: pass 
            
            time.sleep(0.005)
    cap.release()

def inference_worker(core_id, model_path, q_in, q_out, stop_event):
    try: psutil.Process().cpu_affinity([core_id]) 
    except: pass
    
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    
    input_shape = interpreter.get_input_details()[0]['shape']
    MODEL_H, MODEL_W = input_shape[1], input_shape[2]

    while not stop_event.is_set():
        try:
            frame, f_id, t_cap = q_in.get(timeout=0.5)
        except queue.Empty:
            continue
            
        h_orig, w_orig = frame.shape[:2]
        t_inf_s = time.perf_counter()
        
        blob = (cv2.resize(frame, (MODEL_W, MODEL_H)).astype(np.float32) / 255.0 * 255 - 128).astype(np.int8)
        
        interpreter.set_tensor(in_idx, np.expand_dims(blob, axis=0))
        interpreter.invoke()
        
        output = interpreter.get_tensor(out_idx)[0].transpose()
        
        boxes, scores = [], []
        for row in output:
            score = (row[4] + 128) / 255.0
            if score > 0.5:
                cx, cy, w, h = (row[:4] + 128) / 255.0 * MODEL_W
                x1 = int((cx - w/2) * w_orig / MODEL_W)
                y1 = int((cy - h/2) * h_orig / MODEL_H)
                boxes.append([x1, y1, int(x1 + w*w_orig/MODEL_W), int(y1 + h*h_orig/MODEL_H)])
                scores.append(score)
        
        indices = nms_numpy(boxes, scores)
        t_inf_e = time.perf_counter()
        
        try:
            q_out.put((f_id, frame, boxes, scores, indices, t_cap, t_inf_s, t_inf_e), block=False)
        except queue.Full:
            pass

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    
    try: psutil.Process().cpu_affinity([3]) 
    except: pass

    q_in_1 = mp.Queue(maxsize=args.queue_size)
    q_in_2 = mp.Queue(maxsize=args.queue_size)
    q_out = mp.Queue(maxsize=10) 
    
    stop_event = mp.Event()
    tel = Telemetry(name="Realtime Pool (No Deadlock)")

    processes = [
        mp.Process(target=capture_worker, args=(q_in_1, q_in_2, stop_event)),
        mp.Process(target=inference_worker, args=(1, args.model, q_in_1, q_out, stop_event)),
        mp.Process(target=inference_worker, args=(2, args.model, q_in_2, q_out, stop_event)),
    ]
    for p in processes: p.start()

    last_displayed_id = -1 

    try:
        while True:
            try:
                data = q_out.get(timeout=0.01)
                f_id, frame, boxes, scores, indices, t_cap, t_inf_s, t_inf_e = data
                
                if f_id > last_displayed_id:
                    last_displayed_id = f_id
                    t_disp_s = time.perf_counter()
                    
                    for i in indices:
                        b, s = boxes[i], scores[i]
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"{s:.2f}", (b[0], b[1]-5), 0, 0.6, (0, 255, 0), 2)

                    tel.draw_ui(frame)
                    cv2.imshow("Realtime AI Pipeline", frame)
                    
                    t_now = time.perf_counter()
                    
                    tel.update((t_inf_s - t_cap)*1000, (t_inf_e - t_inf_s)*1000, 
                               (t_now - t_disp_s)*1000, (t_now - t_cap)*1000)

            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("\nĐang dọn dẹp hệ thống...")
        stop_event.set()
        for p in processes: p.join(timeout=1)
        for p in processes: 
            if p.is_alive(): p.terminate()
        cv2.destroyAllWindows()
