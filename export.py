from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/segment/roadrunner_only/weights/best.pt')

# Export to INT8 TFLite with proper quantization
model.export(
    format='tflite',
    imgsz=640,
    int8=True,  # Enable INT8 quantization
    data='roadrunner.yaml',  # Needed for calibration
    name='exported_model_int8'  # Specify output name
)