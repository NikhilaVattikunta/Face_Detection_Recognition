import streamlit as st
import cv2
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Face Recognition System", layout="wide")

st.title("🎯 Face Recognition & Attendance System")

# Sidebar
st.sidebar.title("Controls")
start = st.sidebar.button("▶ Start Camera")
stop = st.sidebar.button("⏹ Stop Camera")

FRAME_WINDOW = st.image([])
status = st.empty()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Attendance file
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

run = False

if start:
    run = True
    status.success("Camera Started")

if stop:
    run = False
    status.warning("Camera Stopped")

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()

    if not ret:
        status.error("Camera not working")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            (x1,y1,x2,y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (100,100))
            face_flat = face.flatten().reshape(1, -1)

            distances, _ = model.kneighbors(face_flat)

            if distances[0][0] > 3600:
                name = "Unknown"
                color = (0,0,255)
            else:
                name = model.predict(face_flat)[0]
                color = (0,255,0)

                # Save attendance
                with open(attendance_file, "a") as f:
                    f.write(f"{name},{pd.Timestamp.now()}\n")

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()

# 📊 Attendance Section
st.subheader("📊 Attendance Records")

if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Attendance",
        data=csv,
        file_name="attendance.csv",
        mime='text/csv'
    )
else:
    st.info("No attendance data yet")
