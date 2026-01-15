from ultralytics import YOLO

dataset_name = "rocket_dataset"
model_name = f"yolo11m_ep20"
model_path = f"train_{dataset_name}/{model_name}/weights/best.pt"
model = YOLO(model_path)

model.train(
    data=f"datasets/{dataset_name}.yaml",
    epochs=20,
    imgsz=1024,
    device=[0,1],
    project=f"runs/train_{dataset_name}",
    name=model_name
)