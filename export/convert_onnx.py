import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.deepshield_model import DeepShieldEngine

def export_calibrated_onnx():
    output_path = "export/deepshield.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = DeepShieldEngine()
    checkpoint_path = "checkpoints/deepshield_calibrated.pth"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f" Loaded calibrated weights from {checkpoint_path}")
    else:
        print("⚠️ Warning: Calibration checkpoint not found. Using initial weights.")
    
    model.eval()

    dummy_spatial = torch.randn(1, 5, 3, 224, 224)
    dummy_freq = torch.randn(1, 5, 3, 224, 224)
    dummy_landmarks = torch.randn(1, 5, 468, 3)
    dummy_global = torch.randn(1, 5, 3, 224, 224)

    print(" Converting calibrated model to ONNX computational graph...")
    torch.onnx.export(
        model,
        (dummy_spatial, dummy_freq, dummy_landmarks, dummy_global),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['spatial', 'freq', 'landmarks', 'global_img'],
        output_names=['probabilities'],
        dynamo=False
    )
    print(f"✅ ONNX model successfully saved to: {output_path}")

if __name__ == "__main__":
    export_calibrated_onnx()