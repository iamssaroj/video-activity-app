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

    # Sample a frame every 0.5 seconds for better activity coverage
    sampling_rate = max(1, fps // 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sampling_rate == 0:
            frame = cv2.resize(frame, (640, 480))  # Resize for consistency
            frames.append(frame)
        count += 1
    cap.release()

    st.write(f"✅ Extracted {len(frames)} key frames")

    # Use a more accurate YOLOv8 model (medium version)
    model = YOLO("yolov8m.pt")

    all_labels = []

    progress_bar = st.progress(0)
    for i, frame in enumerate(frames):
        result = model(frame)[0]

        # Filter predictions by confidence threshold (0.5)
        labels = [
            model.names[int(cls)]
            for cls, conf in zip(result.boxes.cls, result.boxes.conf)
            if conf > 0.5
        ]
        all_labels.extend(labels)
        progress_bar.progress((i + 1) / len(frames))

    summary = Counter(all_labels).most_common()

    st.subheader("📝 Activity Summary")
    if summary:
        for label, count in summary:
            st.write(f"• **{label.capitalize()}** detected **{count}** times")
    else:
        st.write("No confident objects detected.")
