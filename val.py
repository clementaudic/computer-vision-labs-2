from ultralytics import YOLO

model = YOLO("all_classes/yolo11n_img1024_ep20/weights/best.pt")
# model = YOLO("runs/detect/train/weights/best.pt")
model.val(data="five_objects_dataset.yaml", save_json=True, save_hybrid=True, name="val_yolo11n_img1024_ep20_five_objects_dataset")