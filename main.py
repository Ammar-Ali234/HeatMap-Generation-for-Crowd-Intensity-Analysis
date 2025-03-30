import cv2
from super_gradients.training import models

model = models.get("yolo_nas_l", pretrained_weights="coco")

Cap = cv2.VideoCapture("video/test.mp4")
vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

while Cap.isOpened():
    ret, frame = Cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cls_id = int(box.cls[0])  # Get class ID
            
            if cls_id in vehicle_classes:  # Filter only vehicle classes
                label = vehicle_classes[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

Cap.release()
cv2.destroyAllWindows()