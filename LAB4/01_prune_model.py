import torch
import torch.nn as nn
from ultralytics import YOLO

# ===== CONFIG =====
WEIGHTS_IN = "best.pt"
WEIGHTS_OUT = "pruned.pt"
PRUNE_RATIO = 0.4

# ===== HAM PRUNE CHO 1 LOP CONV + BN =====
def prune_conv_safe(conv, bn, ratio):
    if not isinstance(conv, nn.Conv2d):
        return 0
    out_ch = conv.out_channels
    n_prune = int(out_ch * ratio)
    if out_ch - n_prune < 8:
        return 0
        
    # ===== TINH DO QUAN TRONG =====
    importance = bn.weight.abs().detach()
    idx = torch.argsort(importance)[:n_prune]
    
    mask = torch.ones(out_ch, dtype=torch.bool)
    mask[idx] = False
    
    # ===== PRUNE CONV =====
    conv.weight.data = conv.weight.data[mask, :, :, :]
    conv.out_channels = conv.weight.shape[0]
    
    # ===== PRUNE BATCHNORM =====
    bn.weight.data = bn.weight.data[mask]
    bn.bias.data = bn.bias.data[mask]
    bn.running_mean = bn.running_mean[mask]
    bn.running_var = bn.running_var[mask]
    
    return n_prune

# ===== HAM MAIN =====
def main():
    print("Load model...")
    y = YOLO(WEIGHTS_IN)
    model = y.model.cpu()
    total_pruned = 0
    
    print("Pruning backbone + neck (safe)...")
    for m in model.modules():
        # CASE 1: layer co conv + bn
        if hasattr(m, "conv") and hasattr(m, "bn"):
            total_pruned += prune_conv_safe(m.conv, m.bn, PRUNE_RATIO)
        # CASE 2: block C2f
        if m.__class__.__name__ == "C2f":
            for sub in [m.cv1, m.cv2]:
                total_pruned += prune_conv_safe(sub.conv, sub.bn, PRUNE_RATIO)
            for b in m.m:
                for sub in [b.cv1, b.cv2]:
                    total_pruned += prune_conv_safe(sub.conv, sub.bn, PRUNE_RATIO)
                    
    print(f"Total pruned channels: {total_pruned}")
    
    # ===== LUU MODEL =====
    y.save(WEIGHTS_OUT)
    print("Saved:", WEIGHTS_OUT)

if __name__ == "__main__":
    main()
