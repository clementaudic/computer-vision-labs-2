from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data="datasets/rocket_dataset.yaml", epochs=20, imgsz=1024, device=[0,1])
