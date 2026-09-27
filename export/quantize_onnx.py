import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model():
    input_onnx = "export/deepshield.onnx"
    output_quantized = "export/deepshield_quantized.onnx"

    if not os.path.exists(input_onnx):
        raise FileNotFoundError(f"Source model '{input_onnx}' not found. Run export/convert_onnx.py first.")

    print("⚡ Applying INT8 Dynamic Quantization to ONNX model...")
    
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_quantized,
        weight_type=QuantType.QUInt8
    )

    print(f"✅ INT8 Quantized ONNX model successfully saved to: {output_quantized}")

if __name__ == "__main__":
    quantize_model()