import torch
from ultralytics import YOLO

MODEL_ORG = "best.pt"
MODEL_PRUNED = "pruned.pt"

FLOPS_ORIGINAL = 8.2   # GFLOPs

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    print("COMPUTE APPROX FLOPs")
    
    y1 = YOLO(MODEL_ORG)
    y2 = YOLO(MODEL_PRUNED)
    
    m1 = y1.model.cpu()
    m2 = y2.model.cpu()
    
    p1 = count_params(m1)
    p2 = count_params(m2)
    
    ratio = p2 / p1
    flops_new = FLOPS_ORIGINAL * ratio
    
    print("\n===== FLOPs ESTIMATION =====")
    print(f"Original FLOPs: {FLOPS_ORIGINAL:.2f} GFLOPs")
    print(f"Pruned FLOPs  : {flops_new:.2f} GFLOPs")
    print(f"Reduction     : {(1 - ratio)*100:.2f}%")

if __name__ == "__main__":
    main()
