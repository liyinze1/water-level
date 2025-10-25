from ultralytics import YOLO

# Load a model
model = YOLO("runs/segment/river-yolo11n-seg-1kep-bs64/weights/best.pt")

# Validate the model
metrics = model.val(data='water_test.txt', plots=True, name='river-model-val-new-photo', task='test')
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category