from ultralytics import YOLO
import wandb

wandb.init(project="water-level")
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="water.yaml", epochs=300, imgsz=640, device=[0, 1], name="yolo11n-seg-300ep")