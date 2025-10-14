# Quantization

This folder performs quantization of the YOLO model used for water detection.

- dataset.py: basic dataset/dataloader implementation used by static quantization
- quantize-static-onnx.py: static quantization
- quantize-dynamic-onnx.py: dynamic quantization
- quantize-qat-onnx.py: QAT (quantization-aware training)
