import torch
from models.model_zoo import build_model

def test_spm():
    B = 2
    T = 16
    N = 64
    num_classes = 40
    
    print("Building model...")
    model = build_model("spm", num_classes=num_classes)
    
    print("Creating dummy input...")
    # [B, T, N, 3] -> 16 slices of 64 points
    x = torch.randn(B, T, N, 3)
    
    print("Running forward pass...")
    logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (B, num_classes)
    print("Success!")

if __name__ == "__main__":
    test_spm()
