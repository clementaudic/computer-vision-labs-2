import os
from ultralytics import YOLO

train_dataset_name = "all_objects_dataset"
model_name = f"yolo11n_img1024_ep20"
model_path = f"train_{train_dataset_name}/{model_name}/weights/best.pt"
model = YOLO(model_path)

val_dataset_name = "five_objects_dataset"

model.val(
    data=f"datasets/{val_dataset_name}.yaml",
    save_json=True,
    save_hybrid=True,
    project=f"runs/val_{val_dataset_name}",
    name=model_name
)