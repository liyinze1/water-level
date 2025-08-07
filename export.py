from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("runs/segment/yolo11n-seg-300ep/weights/best.pt")

# Export the model to TFLite format
model.export(format="tflite")