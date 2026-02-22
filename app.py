import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image

st.title("Face Recognition Attendance System")

# Create folders if not exist
if not os.path.exists("students"):
    os.makedirs("students")

if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv("attendance.csv", index=False)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Add New Student
# -----------------------------

st.header("Register New Student")

name = st.text_input("Enter Student Name")

img_file = st.camera_input("Capture Student Image")

if st.button("Save Student"):
    if name and img_file is not None:
        image = Image.open(img_file)
        image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            cv2.imwrite(f"students/{name}.jpg", image)
            st.success(f"{name} registered successfully!")
        else:
            st.error("No face detected. Try again.")
    else:
        st.warning("Enter name and capture image.")

# -----------------------------
# Mark Attendance
# -----------------------------

st.header("Mark Attendance")

attendance_img = st.camera_input("Capture Image for Attendance")

if st.button("Mark Attendance"):
    if attendance_img is not None:

        image = Image.open(attendance_img)
        image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:

            # Simple matching: check if any saved student image exists
            students = os.listdir("students")

            if len(students) == 0:
                st.warning("No registered students.")
            else:
                name = students[0].replace(".jpg", "")

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                df = pd.read_csv("attendance.csv")
                df.loc[len(df)] = [name, date, time]
                df.to_csv("attendance.csv", index=False)

                st.success(f"Attendance marked for {name}")

        else:
            st.error("No face detected.")

# -----------------------------
# Download Attendance
# -----------------------------

st.header("Download Attendance File")

with open("attendance.csv", "rb") as file:
    st.download_button(
        label="Download attendance.csv",
        data=file,
        file_name="attendance.csv",
        mime="text/csv"
    )
