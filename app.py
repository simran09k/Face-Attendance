
import streamlit as st
import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import os

st.title("🎓 Face Recognition Attendance System")

if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

def mark_attendance(name):
    try:
        df = pd.read_csv("attendance.csv")
    except:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        now = datetime.now()
        new_entry = {
            "Name": name,
            "Date": today,
            "Time": now.strftime("%H:%M:%S")
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv("attendance.csv", index=False)
        st.success(f"{name} marked present!")

st.header("📷 Start Attendance")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not working")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
                mark_attendance(name)
            else:
                name = "Unknown"
        else:
            name = "No Data"

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()

if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv")
    st.dataframe(df)
