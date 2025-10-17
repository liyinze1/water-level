import os
import numpy as np

import onnx
from torchvision.transforms import v2
import onnxruntime as ort

from dataset import WaterLevelDataset

#
# definitions
#
base_folder = os.path.dirname(__file__)
imgsz = 640  # the default on train.py
onnx_model_path = os.path.join(base_folder, "../runs/segment/yolo11l-seg-300ep/weights/best.onnx")
config_file = os.path.join(base_folder, "../water.yaml")

transform = v2.Compose([
    v2.Resize([imgsz, imgsz]),  # we need to resize both h and w
    v2.ToTensor()
])

dataset = WaterLevelDataset(config_path=config_file, transform=transform)
image, labels = dataset[0]


print(f"Loading {onnx_model_path}")
model = onnx.load(onnx_model_path)
onnx.checker.check_model(model)

ort_sess = ort.InferenceSession(onnx_model_path)
outputs = ort_sess.run(None, {'images': np.expand_dims(image.numpy(), 0)})
# outputs = [boxes, masks]

# TODO: look at val.ipynb + plot.py and extract from the code the mask of the GT and Predicted.
# Calculate 
# - torchmetrics.detection.mean_ap.MeanAveragePrecision ==> mAP and mAR
# 