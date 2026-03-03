from onnxruntime.quantization import quantize_dynamic, QuantType

# Path to the original ONNX model
onnx_model_path = "./model/best.onnx"

# Path to save the quantized ONNX model
quantized_model_path = "./model/best_int8.onnx"

# Perform dynamic quantization
quantize_dynamic(
    model_input=onnx_model_path,       # Path to the FP32 model
    model_output=quantized_model_path, # Path to save the INT8 model
    weight_type=QuantType.QInt8        # Quantize weights to INT8
)

print(f"Dynamic quantization complete. Quantized model saved to {quantized_model_path}")