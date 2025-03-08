import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLOv8 gender detection model
model = YOLO("yolov8l.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define class labels (assuming 0 for female, 1 for male)
class_labels = ["Female", "Male"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform gender detection
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID

            label = f"{class_labels[class_id]}: {confidence:.2f}"
            color = (0, 255, 0) if class_id == 1 else (255, 0, 0)  # Blue for Female, Green for Male

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the output
    cv2.imshow("Real-Time Gender Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
