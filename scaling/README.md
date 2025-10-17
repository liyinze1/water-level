# Quantization

This folder performs quantization of the YOLO model used for water detection.

- dataset.py: basic dataset/dataloader implementation used by static quantization
- quantize-static-onnx.py: static quantization
- quantize-dynamic-onnx.py: dynamic quantization
- quantize-qat-onnx.py: QAT (quantization-aware training)


## TODO:

- QAT
- validation (val.py) that computes mAP and mAR on validation set using models

## Refs

- [Neural Compressor](https://github.com/onnx/neural-compressor)
- [Lightnet](https://eavise.gitlab.io/lightnet/index.html)
