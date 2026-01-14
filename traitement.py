import cv2
from ultralytics import YOLO
from sys import argv

# Read train number from command-line (default to '7' if not provided)
train_num = argv[1] if len(argv) > 1 else '8'
# argv[2].endswith(".mp4")
video_file = f"{argv[2]}" if len(argv) > 2 and argv[2] != '0' else 0
# model_path = f"runs/detect/train{train_num}/weights/best.pt"
model_path = "../all_classes/yolo11m_img640_ep20/weights/best.pt"

# model_path = "/home/bezmaternykh/Bureau/Deep Learning/all_classes/yolo11n_img1024_ep20/weights/best.pt"

cap = cv2.VideoCapture(video_file)
cv2.namedWindow('Détection', cv2.WINDOW_NORMAL)

model = YOLO(model_path)

# Optional output file: provide an output path as 3rd CLI arg (argv[3])
save_output = f"videos/{argv[3]}" if len(argv) > 3 else None
writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
 
    if save_output and writer is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output, fourcc, fps, (w, h))

    image = model.predict(source=frame, show=False, conf=0.5, save=False, save_txt=False).pop()
    image_with_boxes = frame.copy()

    for box in image.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        c1 = int(cls_id * 255/17)
        c2 = int(255 - cls_id * 255/17)
        color = (c1, c2, 0)
        cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 1)

    cv2.imshow('Détection', image_with_boxes)

    if writer is not None:
        writer.write(image_with_boxes)

    # Attendre 30 ms entre les frames et quitter si 'q' est pressé
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release writer if used
if writer is not None:
    writer.release()

cap.release()
cv2.destroyAllWindows()