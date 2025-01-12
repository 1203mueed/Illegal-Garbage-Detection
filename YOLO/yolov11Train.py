from ultralytics import YOLO

# Load a model
model = YOLO("YOLO11l-seg.pt")

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)