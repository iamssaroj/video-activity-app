import streamlit as st
from ultralytics import YOLO
import tempfile
import imageio

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this model file is accessible

st.title("🔍 Video Activity Detection using YOLOv8")
st.write("Upload a video and get a summary of detected activities.")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Use imageio to read video frames
    reader = imageio.get_reader(tfile.name)
    frame_count = 0
    detected_objects = {}

    stframe = st.empty()
    st.write("Processing... please wait ⏳")

    try:
        for frame in reader:
            frame_count += 1
            if frame_count % 30 == 0:  # process every 30th frame
                results = model(frame)[0]
                for cls in results.boxes.cls:
                    label = model.names[int(cls)]
                    detected_objects[label] = detected_objects.get(label, 0) + 1
    except Exception as e:
        st.error(f"Error processing video frames: {e}")
    finally:
        reader.close()

    # Summary
    st.success("✅ Video processed!")
    st.subheader("📊 Detected Object Summary")
    if detected_objects:
        for obj, count in detected_objects.items():
            st.write(f"🔹 {obj}: {count} times")
    else:
        st.write("No objects detected.")
