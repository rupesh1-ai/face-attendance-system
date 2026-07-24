import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Attendify Vision",
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

if "attendance_mode" not in st.session_state:
    st.session_state.attendance_mode = False

# ---------------------------------
# Title
# ---------------------------------

st.title("🎓 Face Attendance System")
st.caption("AI-powered student attendance system")

st.divider()

# ---------------------------------
# Registered Students
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
        st.session_state.attendance_mode = True
        st.session_state.register_mode = False
        st.rerun()

with control2:

    if st.button(
        "➕ Register Student",
        use_container_width=True
    ):
        st.session_state.register_mode = True
        st.session_state.attendance_mode = False
        st.rerun()

# ---------------------------------
# REGISTER STUDENT
# ---------------------------------

if st.session_state.register_mode:

    st.divider()

    st.subheader("👤 Register New Student")

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name"
    )

    picture = st.camera_input(
        "📸 Take Student Photo",
        key="registration_camera"
    )

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

                    safe_name = "".join(
                        c for c in name
                        if c.isalnum() or c in (" ", "-", "_")
                    ).strip()

                    if not safe_name:

                        st.error(
                            "❌ Please enter a valid student name."
                        )

                    else:

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
                            for file in os.listdir(student_folder)
                            if file.lower().endswith(
                                (".jpg", ".jpeg", ".png")
                            )
                        ]

                        image_number = len(existing_images) + 1

                        face_path = os.path.join(
                            student_folder,
                            f"{image_number}.jpg"
                        )

                        try:

                            with open(face_path, "wb") as file:
                                file.write(
                                    picture.getvalue()
                                )

                            st.success(
                                f"✅ {safe_name} registered successfully!"
                            )

                            st.session_state.register_mode = False

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"❌ Could not save photo: {e}"
                            )

        with cancel_col:

            if st.button(
                "❌ Cancel",
                use_container_width=True
            ):

                st.session_state.register_mode = False
                st.rerun()

    else:

        if st.button("❌ Cancel Registration"):

            st.session_state.register_mode = False
            st.rerun()

# ---------------------------------
# ATTENDANCE CAMERA
# ---------------------------------

if st.session_state.attendance_mode:

    st.divider()

    st.subheader("🎥 Take Attendance")

    st.info(
        "Position one student clearly in front of the camera."
    )

    attendance_photo = st.camera_input(
        "📷 Capture Student",
        key="attendance_camera"
    )

    if attendance_photo is not None:

        st.image(
            attendance_photo,
            caption="Attendance Photo",
            width=350
        )

        st.success(
            "📸 Photo captured successfully."
        )

        st.info(
            "Face recognition will be connected after we confirm the camera system works correctly."
        )

    if st.button(
        "❌ Close Attendance Camera"
    ):

        st.session_state.attendance_mode = False
        st.rerun()

# ---------------------------------
# Today's Attendance
# ---------------------------------

st.divider()

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

# ---------------------------------
# Attendance History
# ---------------------------------

st.divider()

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