import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from datetime import datetime

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide"
)

ATTENDANCE_FILE = "attendance.csv"
FACES_FOLDER = "faces"

os.makedirs(FACES_FOLDER, exist_ok=True)

# ---------------------------------
# Session State
# ---------------------------------

if "register_mode" not in st.session_state:
    st.session_state.register_mode = False

# ---------------------------------
# Title
# ---------------------------------

st.title("🎓 Face Attendance System")
st.caption("AI-powered student attendance using face recognition")

st.divider()

# ---------------------------------
# Count Registered Students
# ---------------------------------

registered_students = len([
    folder
    for folder in os.listdir(FACES_FOLDER)
    if os.path.isdir(os.path.join(FACES_FOLDER, folder))
])

# ---------------------------------
# Load Attendance
# ---------------------------------

columns = ["Name", "Date", "Time", "Status"]

if os.path.exists(ATTENDANCE_FILE):
    try:
        attendance = pd.read_csv(ATTENDANCE_FILE)

        # Make sure required columns exist
        for column in columns:
            if column not in attendance.columns:
                attendance[column] = ""

        attendance = attendance[columns]

    except Exception:
        attendance = pd.DataFrame(columns=columns)
else:
    attendance = pd.DataFrame(columns=columns)

# ---------------------------------
# Today's Attendance
# ---------------------------------

today = datetime.now().strftime("%d-%m-%Y")

if not attendance.empty:

    today_attendance = attendance[
        attendance["Date"].astype(str) == today
    ]

    today_unique = today_attendance.drop_duplicates(
        subset=["Name"]
    )

else:
    today_unique = attendance

# ---------------------------------
# Dashboard Statistics
# ---------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👥 Registered Students",
        registered_students
    )

with col2:
    st.metric(
        "✅ Present Today",
        len(today_unique)
    )

with col3:

    if registered_students > 0:
        attendance_percentage = (
            len(today_unique) / registered_students
        ) * 100

        attendance_percentage = min(
            attendance_percentage,
            100
        )

    else:
        attendance_percentage = 0

    st.metric(
        "📊 Attendance Rate",
        f"{attendance_percentage:.0f}%"
    )

st.divider()

# ---------------------------------
# Attendance Control
# ---------------------------------

st.subheader("🎯 Attendance Control")

control1, control2 = st.columns(2)

with control1:

    if st.button(
        "🎥 Start Attendance",
        use_container_width=True
    ):
        st.session_state.register_mode = False

        st.info(
            "🎥 Online face recognition will be added next."
        )

with control2:

    if st.button(
        "➕ Register Student",
        use_container_width=True
    ):
        st.session_state.register_mode = True

# ---------------------------------
# Student Registration
# ---------------------------------

if st.session_state.register_mode:

    st.divider()
    st.subheader("👤 Register New Student")

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name"
    )

    picture = st.camera_input(
        "📸 Take Student Photo"
    )

    # ---------------------------------
    # Photo Captured
    # ---------------------------------

    if picture is not None:

        st.image(
            picture,
            caption="Captured Photo",
            width=350
        )

        save_col, cancel_col = st.columns(2)

        with save_col:

            if st.button(
                "✅ Save & Register",
                use_container_width=True
            ):

                name = student_name.strip()

                if not name:

                    st.warning(
                        "⚠️ Please enter the student's name."
                    )

                else:

                    # Clean student name
                    safe_name = "".join(
                        c for c in name
                        if c.isalnum() or c in (" ", "-", "_")
                    ).strip()

                    if not safe_name:

                        st.error(
                            "❌ Please enter a valid student name."
                        )

                    else:

                        # Convert camera image to OpenCV image
                        file_bytes = np.frombuffer(
                            picture.getvalue(),
                            dtype=np.uint8
                        )

                        image = cv2.imdecode(
                            file_bytes,
                            cv2.IMREAD_COLOR
                        )

                        if image is None:

                            st.error(
                                "❌ Could not read the captured image."
                            )

                        else:

                            # Load OpenCV face detector
                            cascade_path = (
                                cv2.data.haarcascades +
                                "haarcascade_frontalface_default.xml"
                            )

                            face_detector = cv2.CascadeClassifier(
                                cascade_path
                            )

                            if face_detector.empty():

                                st.error(
                                    "❌ Face detector could not be loaded."
                                )

                            else:

                                gray = cv2.cvtColor(
                                    image,
                                    cv2.COLOR_BGR2GRAY
                                )

                                faces = face_detector.detectMultiScale(
                                    gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(80, 80)
                                )

                                # -------------------------
                                # Face Validation
                                # -------------------------

                                if len(faces) == 0:

                                    st.error(
                                        "❌ No face detected. "
                                        "Take another photo with your "
                                        "face clearly visible."
                                    )

                                elif len(faces) > 1:

                                    st.error(
                                        "❌ Multiple faces detected. "
                                        "Only one student should be visible."
                                    )

                                else:

                                    x, y, w, h = faces[0]

                                    face = gray[
                                        y:y + h,
                                        x:x + w
                                    ]

                                    face = cv2.resize(
                                        face,
                                        (200, 200)
                                    )

                                    # -------------------------
                                    # Student Folder
                                    # -------------------------

                                    student_folder = os.path.join(
                                        FACES_FOLDER,
                                        safe_name
                                    )

                                    os.makedirs(
                                        student_folder,
                                        exist_ok=True
                                    )

                                    existing_images = [
                                        file
                                        for file in os.listdir(
                                            student_folder
                                        )
                                        if file.lower().endswith(
                                            (
                                                ".jpg",
                                                ".jpeg",
                                                ".png"
                                            )
                                        )
                                    ]

                                    image_number = (
                                        len(existing_images) + 1
                                    )

                                    face_path = os.path.join(
                                        student_folder,
                                        f"{image_number}.jpg"
                                    )

                                    success = cv2.imwrite(
                                        face_path,
                                        face
                                    )

                                    if success:

                                        st.success(
                                            f"✅ {safe_name} "
                                            "registered successfully!"
                                        )

                                        st.session_state.register_mode = False

                                        st.rerun()

                                    else:

                                        st.error(
                                            "❌ Could not save "
                                            "the face image."
                                        )

        with cancel_col:

            if st.button(
                "❌ Cancel Registration",
                use_container_width=True
            ):
                st.session_state.register_mode = False
                st.rerun()

    # ---------------------------------
    # No Photo Yet
    # ---------------------------------

    else:

        if st.button(
            "❌ Cancel Registration",
            use_container_width=True
        ):
            st.session_state.register_mode = False
            st.rerun()

st.divider()

# ---------------------------------
# Today's Attendance
# ---------------------------------

st.subheader("📅 Today's Attendance")

if today_unique.empty:

    st.info(
        "No attendance has been recorded today."
    )

else:

    st.dataframe(
        today_unique,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ---------------------------------
# Attendance History
# ---------------------------------

st.subheader("📚 Attendance History")

if attendance.empty:

    st.info(
        "No attendance records available."
    )

else:

    st.dataframe(
        attendance,
        use_container_width=True,
        hide_index=True
    )