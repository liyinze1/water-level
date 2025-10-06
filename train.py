from ultralytics import YOLO
import wandb
import os

os.system('export MKL_THREADING_LAYER=GNU')

wandb.init(project="water-level")

model = YOLO("yolo11n-seg.pt")
#model = YOLO("yolo11l-seg.pt") 

# Train the model with 2 GPUs
results = model.train(data="water.yaml", epochs=300, imgsz=640, device=[0, 1], name="yolo11l-seg-300ep")