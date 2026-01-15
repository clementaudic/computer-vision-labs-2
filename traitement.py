from sys import argv
import cv2
from ultralytics import YOLO

video_file = f"{argv[1]}" if len(argv) > 1 else 0
# model_path = "runs/detect/train/weights/best.pt"
model_path = "all_classes/yolo11n_img1024_ep20/weights/best.pt"

cap = cv2.VideoCapture(video_file)
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

model = YOLO(model_path)

# Optional output file: provide an output path as 2nd CLI arg (argv[2])
save_output = f"videos/{argv[2]}" if len(argv) >= 3 else None
writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # continue
        break
 
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

    cv2.imshow('Detection', image_with_boxes)

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