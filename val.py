from ultralytics import YOLO

# Load a model
model = YOLO("runs/segment/yolo11n-seg-300ep/weights/best.pt")  # load an official model

# Validate the model
metrics = model.val(data='river.yaml', plots=True, name='water-model-val-river')  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category