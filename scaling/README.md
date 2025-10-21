# Quantization

This folder performs quantization of the YOLO model used for water detection.

- dataset.py: basic dataset/dataloader implementation used by static quantization
- quantize-static-onnx.py: static quantization
- quantize-dynamic-onnx.py: dynamic quantization
- quantize-qat-onnx.py: QAT (quantization-aware training)


TODO: read [TODO.md](TODO.md)


## Important remarks:

Converting ONNX model to TFLite model is not feasible (so far) using Python 3.13 because several dependency libraries were not migrated to this version, such as onnx-tf, onnx2tf, ai_edge_litert, etc.

## Refs

- [Neural Compressor](https://github.com/onnx/neural-compressor)
- [Lightnet](https://eavise.gitlab.io/lightnet/index.html)
