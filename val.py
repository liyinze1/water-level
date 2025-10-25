from ultralytics import YOLO

# Load a model
model = YOLO("runs/segment/waterv2_roadrunner/weights/best.pt")

# Validate the model
metrics = model.val(data='water.yaml', plots=True, name='waterv2_roadrunner-test', split='test')
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category