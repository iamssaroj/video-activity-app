import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from collections import Counter
import numpy as np

# --- App Title ---
st.set_page_config(page_title="🎬 VidGist – Smart Video Summarization Tool", layout="wide")
st.title("🎥 AI-Powered Video Activity Detector")
st.markdown("Upload a video and get a **summary of detected objects and activities** using YOLOv8.")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("YOLO Model", ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"], index=1)
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
sampling_option = st.sidebar.selectbox("Frame Sampling Interval", ["Every 0.5 sec", "Every 1 sec", "Every 2 sec"], index=1)

# --- Upload File ---
uploaded_file = st.file_uploader("📤 Upload your video", type=["mp4", "avi", "mov"])

if uploaded_file:
    if st.button("🔍 Analyze Video"):
        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine frame sampling interval
        if sampling_option == "Every 0.5 sec":
            sampling_rate = max(1, fps // 2)
        elif sampling_option == "Every 1 sec":
            sampling_rate = fps
        else:
            sampling_rate = fps * 2

        frames = []
        count = 0
        with st.spinner("Extracting frames..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % sampling_rate == 0:
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
                count += 1
            cap.release()

        st.success(f"✅ Extracted {len(frames)} key frames")

        # Load selected YOLO model
        model = YOLO(model_choice)
        all_labels = []

        st.subheader("📦 Detecting Activities...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, frame in enumerate(frames):
            result = model(frame)[0]
            labels = [
                model.names[int(cls)]
                for cls, conf in zip(result.boxes.cls, result.boxes.conf)
                if conf > confidence_threshold
            ]
            all_labels.extend(labels)
            progress_bar.progress((i + 1) / len(frames))
            status_text.text(f"Processed frame {i+1} of {len(frames)}")

        summary = Counter(all_labels).most_common()

        st.subheader("📝 Activity Summary")
        if summary:
            with st.expander("🔽 View Detected Object Summary"):
                for label, count in summary:
                    st.markdown(f"- **{label.capitalize()}**: {count} times")
        else:
            st.warning("No confident objects detected in the video.")
