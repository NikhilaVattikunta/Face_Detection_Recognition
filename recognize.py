import cv2
import pickle
import numpy as np
import csv
from datetime import datetime
import os

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

attendance_file = "attendance.csv"

# Create file if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

def mark_attendance(name):
    with open(attendance_file, "r+") as f:
        data = f.readlines()
        names = [line.split(",")[0] for line in data]

        if name not in names and name != "Unknown":
            writer = csv.writer(f)
            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# ✅ Stable camera setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

while True:
    ret, frame = cap.read()

    # ✅ Do not break if frame missing
    if not ret:
        print("⚠️ Camera frame not received, retrying...")
        continue

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # SAME preprocessing as training
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (100, 100))
            face_flat = face.flatten().reshape(1, -1)

            # Get distance
            distances, indices = model.kneighbors(face_flat)

            print("Distance:", distances[0][0])  # Debug

            # ✅ Adjusted threshold based on your data
            if distances[0][0] > 3600:
                name = "Unknown"
                color = (0, 0, 255)  # Red
            else:
                name = model.predict(face_flat)[0]
                color = (0, 255, 0)  # Green

                mark_attendance(name)

            # Draw box + name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, 2)

    cv2.imshow("Face Recognition System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
