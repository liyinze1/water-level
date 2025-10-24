from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/segment/river-yolo11n-seg-1kep-bs64/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict('photos-1024/images', save=True, imgsz=640, conf=0.5, name='roadrunner-photo-1024')
# model.predict('./Leuven', save=True, imgsz=640, conf=0.5, name='leuven-photo')
# model.predict('./river.txt', save=True, imgsz=640, conf=0.5, name='river-predict')