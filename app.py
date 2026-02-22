import streamlit as st
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import os
from PIL import Image

st.title("🎓 Face Recognition Attendance System")

# -------------------------
# LOAD ENCODINGS
# -------------------------
if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

# -------------------------
# MARK ATTENDANCE
# -------------------------
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

# -------------------------
# CAMERA INPUT (Cloud Compatible)
# -------------------------
st.header("📷 Take Picture for Attendance")

picture = st.camera_input("Capture Image")

if picture is not None:
    image = Image.open(picture)
    rgb_image = np.array(image)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding in face_encodings:
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
                mark_attendance(name)
                st.success(f"Recognized: {name}")
            else:
                st.error("Unknown Face")
        else:
            st.warning("No student data available")

# -------------------------
# SHOW ATTENDANCE
# -------------------------
st.header("📊 Attendance Records")

if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv")
    st.dataframe(df)
else:
    st.write("No attendance yet.")
