import torch
import os
from ultralytics import YOLO

MODEL_ORG = "best.pt"
MODEL_PRUNED = "pruned.pt"

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_size(path):
    return os.path.getsize(path) / 1e6

def main():
    print("ANALYZE PARAMS & SIZE")
    
    y1 = YOLO(MODEL_ORG)
    y2 = YOLO(MODEL_PRUNED)
    
    m1 = y1.model.cpu()
    m2 = y2.model.cpu()
    
    p1 = count_params(m1)
    p2 = count_params(m2)
    
    s1 = get_size(MODEL_ORG)
    s2 = get_size(MODEL_PRUNED)
    
    print("\n===== PARAMS =====")
    print(f"Original: {p1:,}")
    print(f"Pruned  : {p2:,}")
    print(f"Reduction: {(1 - p2/p1)*100:.2f}%")
    
    print("\n===== SIZE =====")
    print(f"Original: {s1:.2f} MB")
    print(f"Pruned  : {s2:.2f} MB")
    print(f"Reduction: {(1 - s2/s1)*100:.2f}%")
    
    print("\n===== FLOPs (BEST MODEL) =====")
    print("Run the line below to view FLOPs:")
    y1.info(verbose=True)

if __name__ == "__main__":
    main()
