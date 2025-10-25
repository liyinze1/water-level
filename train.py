import torch
from ultralytics import YOLO
import wandb
import os

os.system('export MKL_THREADING_LAYER=GNU')

wandb.init(project="water-level")

model = YOLO("yolo11n-seg.pt")
#model = YOLO("yolo11l-seg.pt") 

# Train the model with 2 GPUs, if available
devices = [0, 1] if torch.cuda.device_count() > 1 else [0]
results = model.train(data="water.yaml", epochs=1000, imgsz=640, device=devices, name="waterv2_roadrunner", batch=64)