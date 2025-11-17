from ultralytics import YOLO
import glob

# Load a model
# model = YOLO("best.pt")

models = glob.glob('runs/segment/roadrunner_only/weights/best_saved_model/*.tflite')

for model_path in models:
    print('*********************************************')
    print(f'Validating model: {model_path}')
    model = YOLO(model_path)
    # Validate the model
    metrics = model.val(data='roadrunner.yaml', plots=True, name='roadrunner-val-' + model_path.split('/')[-1])
    metrics.box.map  # map50-95(B)
    metrics.box.map50  # map50(B)
    metrics.box.map75  # map75(B)
    metrics.box.maps  # a list contains map50-95(B) of each category
    metrics.seg.map  # map50-95(M)
    metrics.seg.map50  # map50(M)
    metrics.seg.map75  # map75(M)
    metrics.seg.maps  # a list contains map50-95(M) of each category