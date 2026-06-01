import torch

try:
    ckpt = torch.load("model/cell_classifier_best.pth", map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")
    else:
        print("No epoch info found in checkpoint.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
