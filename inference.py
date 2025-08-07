from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/segment/yolo11n-seg-300ep/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict('data.txt', save=True, imgsz=640, conf=0.5)