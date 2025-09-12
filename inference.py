from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/segment/yolo11l-seg-300ep/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict('./Leuven', save=True, imgsz=640, conf=0.5, name='11l-predict')