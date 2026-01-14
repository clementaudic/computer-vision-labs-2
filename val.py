from ultralytics import YOLO

model = YOLO("../all_classes/yolo11m_img640_ep20/weights/best.pt")
# model = YOLO("runs/detect/train12/weights/best.pt")
model.val(data="../test_dataset_with_five_objects.yaml", save_json=True, save_hybrid=True)