import streamlit as st
import cv2
import numpy as np
import tempfile
from src.lane_detector import LaneDetector

st.title("🚗 Lane Detection Debug")

@st.cache_resource
def get_detector():
    return LaneDetector()

detector = get_detector()

st.write("Detector loaded successfully:", detector is not None)

uploaded_file = st.file_uploader("Choose video...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    st.info("Processing started...")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("Error: Cannot open video file")
        st.stop()

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1  # avoid division by zero
    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = detector.process_frame(frame)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            stframe.image(result_rgb, use_column_width=True)
        except Exception as e:
            st.error(f"Error in processing frame: {e}")
            break

        processed += 1
        progress_bar.progress(processed / frame_count)

    cap.release()
    st.success("Done!")
