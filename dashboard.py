import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Attendify Vision",
    page_icon="🎓",
    layout="wide"
)

ATTENDANCE_FILE = "attendance.csv"
FACES_FOLDER = "faces"

os.makedirs(FACES_FOLDER, exist_ok=True)

# ==========================================
# SESSION STATE
# ==========================================

if "register_mode" not in st.session_state:
    st.session_state.register_mode = False

if "attendance_mode" not in st.session_state:
    st.session_state.attendance_mode = False

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_registered_students():
    students = []

    if os.path.exists(FACES_FOLDER):
        for item in os.listdir(FACES_FOLDER):

            path = os.path.join(FACES_FOLDER, item)

            if os.path.isdir(path):
                students.append(item)

    return sorted(students)


def load_attendance():

    columns = [
        "Name",
        "Date",
        "Time",
        "Status"
    ]

    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(ATTENDANCE_FILE)

        for column in columns:
            if column not in df.columns:
                df[column] = ""

        return df[columns]

    except Exception:
        return pd.DataFrame(columns=columns)


def mark_attendance(student_name):

    attendance = load_attendance()

    now = datetime.now()

    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%I:%M:%S %p")

    # Check if student already attended today

    if not attendance.empty:

        duplicate = attendance[
            (attendance["Name"].astype(str) == student_name)
            &
            (attendance["Date"].astype(str) == date)
        ]

        if not duplicate.empty:
            return False, "already_marked"

    new_record = pd.DataFrame(
        [{
            "Name": student_name,
            "Date": date,
            "Time": time,
            "Status": "Present"
        }]
    )

    attendance = pd.concat(
        [attendance, new_record],
        ignore_index=True
    )

    attendance.to_csv(
        ATTENDANCE_FILE,
        index=False
    )

    return True, "success"


# ==========================================
# DATA
# ==========================================

registered_student_names = get_registered_students()

registered_students = len(
    registered_student_names
)

attendance = load_attendance()

today = datetime.now().strftime(
    "%d-%m-%Y"
)

if not attendance.empty:

    today_attendance = attendance[
        attendance["Date"].astype(str) == today
    ]

    today_unique = today_attendance.drop_duplicates(
        subset=["Name"]
    )

else:

    today_unique = attendance


# ==========================================
# HEADER
# ==========================================

st.title("🎓 Attendify Vision")

st.caption(
    "Smart Student Attendance Management System"
)

st.divider()


# ==========================================
# DASHBOARD
# ==========================================

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

        percentage = (
            len(today_unique)
            /
            registered_students
        ) * 100

        percentage = min(
            percentage,
            100
        )

    else:

        percentage = 0

    st.metric(
        "📊 Attendance Rate",
        f"{percentage:.0f}%"
    )


st.divider()


# ==========================================
# ATTENDANCE CONTROL
# ==========================================

st.subheader(
    "🎯 Attendance Control"
)

control1, control2 = st.columns(2)


with control1:

    if st.button(
        "🎥 Start Attendance",
        use_container_width=True,
        type="primary"
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


# ==========================================
# REGISTER STUDENT
# ==========================================

if st.session_state.register_mode:

    st.divider()

    st.subheader(
        "👤 Register New Student"
    )

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name"
    )

    registration_photo = st.camera_input(
        "📸 Take Student Photo",
        key="registration_camera"
    )


    if registration_photo is not None:

        st.image(
            registration_photo,
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
                        "⚠️ Enter the student's name first."
                    )

                else:

                    safe_name = "".join(
                        character
                        for character in name
                        if character.isalnum()
                        or character in (
                            " ",
                            "-",
                            "_"
                        )
                    ).strip()


                    if not safe_name:

                        st.error(
                            "❌ Invalid student name."
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

                            for file
                            in os.listdir(
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
                            len(existing_images)
                            + 1
                        )


                        image_path = os.path.join(
                            student_folder,
                            f"{image_number}.jpg"
                        )


                        try:

                            with open(
                                image_path,
                                "wb"
                            ) as file:

                                file.write(
                                    registration_photo.getvalue()
                                )


                            st.success(
                                f"✅ {safe_name} registered successfully!"
                            )

                            st.session_state.register_mode = False

                            st.rerun()


                        except Exception as error:

                            st.error(
                                f"❌ Registration failed: {error}"
                            )


        with cancel_col:

            if st.button(
                "❌ Cancel Registration",
                use_container_width=True
            ):

                st.session_state.register_mode = False

                st.rerun()


# ==========================================
# ATTENDANCE MODE
# ==========================================

if st.session_state.attendance_mode:

    st.divider()

    st.subheader(
        "🎥 Take Attendance"
    )


    if registered_students == 0:

        st.warning(
            "⚠️ No students are registered yet."
        )

        st.info(
            "Register at least one student before taking attendance."
        )


    else:

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


            st.write(
                "### Select Student"
            )


            selected_student = st.selectbox(
                "Registered Student",
                registered_student_names
            )


            if st.button(
                "✅ Mark Attendance",
                use_container_width=True,
                type="primary"
            ):

                success, status = mark_attendance(
                    selected_student
                )


                if success:

                    st.success(
                        f"✅ Attendance marked for {selected_student}!"
                    )

                    st.balloons()

                    st.session_state.attendance_mode = False

                    st.rerun()


                elif status == "already_marked":

                    st.warning(
                        f"⚠️ {selected_student} is already marked present today."
                    )


    if st.button(
        "❌ Close Attendance Camera"
    ):

        st.session_state.attendance_mode = False

        st.rerun()


# ==========================================
# TODAY'S ATTENDANCE
# ==========================================

st.divider()

st.subheader(
    "📅 Today's Attendance"
)

# Reload because attendance may have changed

attendance = load_attendance()

if not attendance.empty:

    today_attendance = attendance[
        attendance["Date"].astype(str)
        == today
    ]

    today_unique = today_attendance.drop_duplicates(
        subset=["Name"]
    )

else:

    today_unique = attendance


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


# ==========================================
# ATTENDANCE HISTORY
# ==========================================

st.divider()

st.subheader(
    "📚 Attendance History"
)

attendance = load_attendance()


if attendance.empty:

    st.info(
        "No attendance records available."
    )

else:

    st.dataframe(
        attendance.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )