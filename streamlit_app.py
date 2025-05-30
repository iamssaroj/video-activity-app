import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt

st.title("🔍 Video Activity Detection using YOLOv8")
st.write("Upload a video and get a summary of detected activities.")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    detected_objects = {}

    stframe = st.empty()
    st.write("Processing... please wait ⏳")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # process every 30th frame
            results = model(frame)[0]
            for result in results.boxes.cls:
                label = model.names[int(result)]
                detected_objects[label] = detected_objects.get(label, 0) + 1

    cap.release()

    # Summary
    st.success("✅ Video processed!")
    st.subheader("📊 Detected Object Summary")
    for obj, count in detected_objects.items():
        st.write(f"🔹 {obj}: {count} times")
