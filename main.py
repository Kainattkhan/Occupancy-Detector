import streamlit as st
import cv2
import pandas as pd
import time
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Title
st.set_page_config(page_title="Occupancy Detection & AC Control", layout="centered")
st.title("Occupancy Detection with YOLOv8 & Auto lights Control")
st.markdown("This app detects people in webcam feed and simulates turning the Lights ON/OFF based on occupancy.")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Logging setup
log_data = []
vacancy_timeout = 3
ac_status = "OFF"
last_seen_time = None

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.ac_status = "OFF"
        self.last_seen_time = None

    def recv(self, frame):
        global log_data

        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.4, verbose=False)
        person_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    person_detected = True
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, tuple(coords[:2]), tuple(coords[2:]), (0, 255, 0), 2)
                    cv2.putText(img, "Person", tuple(coords[:2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()

        if person_detected:
            self.last_seen_time = current_time
            if self.ac_status != "ON":
                self.ac_status = "ON"
                st.info(f"[{timestamp}] Lights ON")
        else:
            if self.last_seen_time and (current_time - self.last_seen_time > vacancy_timeout):
                if self.ac_status != "OFF":
                    self.ac_status = "OFF"
                    st.warning(f"[{timestamp}] Lights OFF")

        log_data.append({
            "timestamp": timestamp,
            "person_detected": int(person_detected),
            "AC_status": self.ac_status
        })

        cv2.putText(img, f"AC: {self.ac_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if self.ac_status == "OFF" else (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream from webcam
# webrtc_streamer(key="occupancy", video_processor_factory=VideoTransformer)
# Stream from webcam
webrtc_streamer(
    key="occupancy",
    video_processor_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


# Show log dataframe
if st.button("📄 Show Log"):
    if log_data:
        df = pd.DataFrame(log_data)
        st.dataframe(df.tail(20))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Log as CSV", csv, "occupancy_log.csv", "text/csv")
    else:
        st.info("No activity logged yet.")
