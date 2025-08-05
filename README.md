#  Occupancy Detection with YOLOv8 & Auto Lights Control

This is a simple Streamlit-based application that detects people using your webcam feed and automatically simulates **turning lights ON or OFF** based on occupancy.

Built using:

* [YOLOv8](https://github.com/ultralytics/ultralytics) for real-time person detection
* Streamlit for the web app interface
* `streamlit-webrtc` to access webcam video

---

## 🚀 Features

* Real-time webcam person detection
* Automatically switches AC/Lights ON when a person is detected
* Turns OFF when no person is detected for a few seconds
* Displays live video with annotations
* Logs activity (timestamp, detection, AC status)
* Option to download activity log as CSV

---

## 📦 Requirements

Install the required Python packages before running:

```bash
pip install streamlit opencv-python pandas ultralytics streamlit-webrtc av
```

---

## 🛠 How it Works

* The app uses the **YOLOv8n** model to detect if a person is in the webcam feed.
* If a person is detected, the system simulates **AC/Lights ON**.
* If no person is detected for more than **3 seconds**, it simulates **AC/Lights OFF**.
* Each detection (or lack of) is timestamped and stored in a log.
* You can view and download this log from the interface.

---

## ▶️ How to Run

Simply run this command in your terminal:

```bash
streamlit run app.py
```

Make sure your webcam is enabled. The app will start in your browser.

---

## 📥 Example Log Format

| timestamp           | person\_detected | AC\_status |
| ------------------- | ---------------- | ---------- |
| 2025-08-05 10:00:01 | 1                | ON         |
| 2025-08-05 10:00:10 | 0                | OFF        |

---

## 📝 Notes

* The app uses the default YOLOv8n model (`yolov8n.pt`).
* You can tweak the confidence threshold and timeout duration inside the code.
* Intended for demo/prototyping purposes, not production.

