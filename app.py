import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import pickle
import os
import face_recognition

st.set_page_config(page_title="Face Attendance System")

st.title("🎓 Face Recognition Attendance System")

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = np.array(data["encodings"])
known_names = np.array(data["names"])

# Create attendance file if not exists
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv("attendance.csv", index=False)

st.header("📸 Capture Image to Mark Attendance")

uploaded_file = st.camera_input("Take a picture")

if st.button("Mark Attendance"):
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = np.array(image)

        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]

            distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )

            best_match_index = np.argmin(distances)

            if distances[best_match_index] < 0.6:
                name = known_names[best_match_index]

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                df = pd.read_csv("attendance.csv")

                # Avoid duplicate entry same day
                if not ((df["Name"] == name) & (df["Date"] == date)).any():
                    df.loc[len(df)] = [name, date, time]
                    df.to_csv("attendance.csv", index=False)
                    st.success(f"✅ Attendance marked for {name}")
                else:
                    st.warning(f"{name} already marked today")

            else:
                st.error("❌ Face not recognized")

        else:
            st.error("No face detected")

# Download attendance
st.header("📥 Download Attendance File")

with open("attendance.csv", "rb") as file:
    st.download_button(
        label="Download attendance.csv",
        data=file,
        file_name="attendance.csv",
        mime="text/csv"
    )
