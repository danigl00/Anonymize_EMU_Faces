import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = "Yunet/Models/yunet_n_dynamic.onnx"
onnx_model = onnx.load(model_path)

# Check the model
onnx.checker.check_model(onnx_model)

# Run inference with onnxruntime
ort_session = ort.InferenceSession(model_path)

# Prepare dummy input data (example)
input_name = ort_session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 480).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {input_name: dummy_input})

print(outputs)
