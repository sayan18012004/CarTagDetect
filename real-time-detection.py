from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(2)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Model
model_path = r"model/for-detecting-license-plates/last.pt"  # Use a raw string to avoid escape sequence issues
model = YOLO(model_path)

# Object classes
classNames = ["License_Plate"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    License_Plate_count = 0

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name == "License_Plate":
                License_Plate_count += 1
                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.6
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"{class_name} {confidence:.2f}", org, font, fontScale, color, thickness)

    # Display the count of ambulances detected
    count_text = f"License Plate detected: {License_Plate_count}"
    cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
