import time, psutil, numpy as np, cv2
 
class Telemetry:
    """
    Class Telemetry: Module giám sát hiệu năng hệ thống (Profiler) theo thời gian thực.
    Đo đạc Throughput (FPS) và phân chia Latency (Cap, Inf, Disp, E2E) để xác định Bottleneck.
    """
    def __init__(self, name="AIoT-Pipeline"):
        self.name = name
        self.color = (0, 255, 0) if "Single" in name else (255, 255, 0)
        self.last_report = time.perf_counter()
        self.count = 0
        self.data = {"cap": [], "inf": [], "disp": [], "e2e": []}
 
    def update(self, cap, inf, disp, e2e):
        self.data["cap"].append(cap)
        self.data["inf"].append(inf)
        self.data["disp"].append(disp)
        self.data["e2e"].append(e2e)
        self.count += 1
        
        now = time.perf_counter()
        elapsed = now - self.last_report
        
        if elapsed >= 1.0:
            n = max(1, len(self.data["cap"]))
            avg = {k: sum(v)/n for k, v in self.data.items()}
            
            fps_e2e = self.count / elapsed
            fps_inf = 1000.0 / avg["inf"] if avg["inf"] > 0 else 0
            
            print(f"\n--- [{self.name}] ---")
            print(f"FPS End-to-End: {fps_e2e:.2f} | FPS Inf: {fps_inf:.2f} | CPU Total: {psutil.cpu_percent()}%")
            print(f"Latency (ms): Cap: {avg['cap']:.2f} | Inf: {avg['inf']:.2f} | Disp: {avg['disp']:.2f} | End-to-End: {avg['e2e']:.2f}")
            
            self.data = {"cap": [], "inf": [], "disp": [], "e2e": []}
            self.count = 0
            self.last_report = now
 
    def draw_ui(self, frame):
        cv2.putText(frame, f"[{self.name}] CPU: {psutil.cpu_percent()}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
 
def nms_numpy(boxes, scores, threshold=0.45):
    """
    Thuật toán Non-Maximum Suppression (NMS) viết hoàn toàn bằng NumPy.
    """
    if len(boxes) == 0: return []
    boxes, scores = np.array(boxes), np.array(scores)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= threshold)[0] + 1]
        
    return keep
