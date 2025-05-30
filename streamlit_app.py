import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from collections import Counter
import numpy as np

st.title("🎬 Video Activity Detection and Summary")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps == 0:
            frames.append(frame)
        count += 1
    cap.release()

    st.write(f"✅ Extracted {len(frames)} key frames")

    model = YOLO("yolov8n.pt")
    all_labels = []

    for frame in frames:
        result = model(frame)[0]
        labels = [model.names[int(cls)] for cls in result.boxes.cls]
        all_labels.extend(labels)

    summary = Counter(all_labels).most_common()
    st.subheader("📝 Activity Summary")
    for label, count in summary:
        st.write(f"• **{label.capitalize()}** detected **{count}** times")
