import streamlit as st
import pandas as pd
import os
import subprocess
import sys
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

# ---------------------------------
# Title
# ---------------------------------

st.title("🎓 Face Attendance System")
st.caption("AI-powered student attendance using face recognition")

st.divider()

# ---------------------------------
# Count Registered Students
# ---------------------------------

registered_students = 0

if os.path.exists(FACES_FOLDER):
    registered_students = len([
        folder
        for folder in os.listdir(FACES_FOLDER)
        if os.path.isdir(os.path.join(FACES_FOLDER, folder))
    ])

# ---------------------------------
# Load Attendance
# ---------------------------------

if os.path.exists(ATTENDANCE_FILE):
    try:
        attendance = pd.read_csv(ATTENDANCE_FILE)
    except Exception:
        attendance = pd.DataFrame(
            columns=["Name", "Date", "Time", "Status"]
        )
else:
    attendance = pd.DataFrame(
        columns=["Name", "Date", "Time", "Status"]
    )

# ---------------------------------
# Today's Attendance
# ---------------------------------

today = datetime.now().strftime("%d-%m-%Y")

if not attendance.empty:

    today_attendance = attendance[
        attendance["Date"].astype(str) == today
    ]

    # Prevent duplicate students affecting statistics
    today_unique = today_attendance.drop_duplicates(
        subset=["Name"]
    )

else:
    today_attendance = attendance
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

# ---------------------------------
# START ATTENDANCE
# ---------------------------------

with control1:

    if st.button(
        "🎥 Start Attendance",
        use_container_width=True
    ):

        with st.spinner(
            "Starting face recognition..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        "face_recognisation.py"
                    ],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:

                    st.success(
                        "✅ Attendance completed successfully!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Face recognition encountered an error."
                    )

                    if result.stderr:
                        st.code(result.stderr)

                    if result.stdout:
                        st.code(result.stdout)

            except Exception as e:

                st.error(
                    f"Unable to start attendance: {e}"
                )

# ---------------------------------
# REGISTER STUDENT BUTTON
# ---------------------------------

with control2:

    if st.button(
        "➕ Register Student",
        use_container_width=True
    ):
        st.session_state["register_mode"] = True

# ---------------------------------
# Student Registration
# ---------------------------------

if st.session_state.get(
    "register_mode",
    False
):

    st.divider()

    st.subheader(
        "👤 Register New Student"
    )

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name"
    )

    register_col1, register_col2 = st.columns(2)

    with register_col1:

        if st.button(
            "📸 Capture Face",
            use_container_width=True
        ):

            if not student_name.strip():

                st.warning(
                    "⚠️ Please enter the student's name."
                )

            else:

                with st.spinner(
                    "Opening camera..."
                ):

                    try:

                        result = subprocess.run(
                            [
                                sys.executable,
                                "register_face.py"
                            ],
                            input=student_name.strip() + "\n",
                            capture_output=True,
                            text=True
                        )

                        if result.returncode == 0:

                            st.success(
                                f"✅ {student_name} registered successfully!"
                            )

                            st.session_state[
                                "register_mode"
                            ] = False

                            st.rerun()

                        else:

                            st.error(
                                "❌ Registration failed."
                            )

                            if result.stderr:
                                st.code(result.stderr)

                            if result.stdout:
                                st.code(result.stdout)

                    except Exception as e:

                        st.error(
                            f"Registration error: {e}"
                        )

    with register_col2:

        if st.button(
            "❌ Cancel Registration",
            use_container_width=True
        ):

            st.session_state[
                "register_mode"
            ] = False

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
        width="stretch",
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
        width="stretch",
        hide_index=True
    )