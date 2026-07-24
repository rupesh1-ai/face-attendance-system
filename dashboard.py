import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Attendify Vision",
    page_icon="🎓",
    layout="wide"
)

ATTENDANCE_FILE = "attendance.csv"
FACES_FOLDER = "faces"

os.makedirs(FACES_FOLDER, exist_ok=True)

# --------------------------------------------------
# LOAD OPENCV SAFELY
# --------------------------------------------------

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception as e:
    cv2 = None
    OPENCV_AVAILABLE = False
    OPENCV_ERROR = str(e)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "register_mode" not in st.session_state:
    st.session_state.register_mode = False

if "attendance_mode" not in st.session_state:
    st.session_state.attendance_mode = False

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_attendance():

    columns = ["Name", "Date", "Time", "Status"]

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


def save_attendance(name):

    attendance = load_attendance()

    today = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    already_present = (
        (attendance["Name"].astype(str) == name) &
        (attendance["Date"].astype(str) == today)
    ).any()

    if already_present:
        return False

    new_record = pd.DataFrame(
        [{
            "Name": name,
            "Date": today,
            "Time": current_time,
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

    return True


def image_to_gray(uploaded_file):

    if not OPENCV_AVAILABLE:
        return None

    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray


def get_face(gray):

    if not OPENCV_AVAILABLE:
        return None

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    face = gray[
        y:y + h,
        x:x + w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    return face


def build_recognizer():

    if not OPENCV_AVAILABLE:
        return None, {}

    training_faces = []
    training_labels = []

    label_to_name = {}

    label = 0

    for student_name in os.listdir(FACES_FOLDER):

        student_folder = os.path.join(
            FACES_FOLDER,
            student_name
        )

        if not os.path.isdir(student_folder):
            continue

        student_added = False

        for filename in os.listdir(student_folder):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                student_folder,
                filename
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            face = get_face(image)

            # Older registration images may already
            # contain only the cropped face.
            if face is None:
                try:
                    face = cv2.resize(
                        image,
                        (200, 200)
                    )
                except Exception:
                    continue

            training_faces.append(face)
            training_labels.append(label)

            student_added = True

        if student_added:

            label_to_name[label] = student_name
            label += 1

    if not training_faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        training_faces,
        np.array(training_labels)
    )

    return recognizer, label_to_name


def recognize_student(uploaded_file):

    gray = image_to_gray(uploaded_file)

    if gray is None:
        return None, None, "Could not read the captured image."

    face = get_face(gray)

    if face is None:
        return (
            None,
            None,
            "Make sure exactly one face is clearly visible."
        )

    recognizer, labels = build_recognizer()

    if recognizer is None:
        return (
            None,
            None,
            "No usable registered student photos were found."
        )

    predicted_label, confidence = recognizer.predict(face)

    name = labels.get(predicted_label)

    # LBPH: lower score means closer match.
    # This is deliberately conservative.
    threshold = 65

    if name is None or confidence > threshold:
        return (
            None,
            confidence,
            "Face not confidently recognized."
        )

    return name, confidence, None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 Attendify Vision")

st.caption(
    "AI-powered student attendance using face recognition"
)

st.divider()

# --------------------------------------------------
# REGISTERED STUDENTS
# --------------------------------------------------

registered_students = len([
    folder
    for folder in os.listdir(FACES_FOLDER)
    if os.path.isdir(
        os.path.join(FACES_FOLDER, folder)
    )
])

attendance = load_attendance()

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

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

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
            len(today_unique) /
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

# --------------------------------------------------
# ATTENDANCE CONTROL
# --------------------------------------------------

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

# --------------------------------------------------
# REGISTER STUDENT
# --------------------------------------------------

if st.session_state.register_mode:

    st.divider()

    st.subheader("👤 Register New Student")

    if not OPENCV_AVAILABLE:

        st.error(
            "Face-processing library could not be loaded."
        )

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
                        "Please enter the student's name."
                    )

                elif not OPENCV_AVAILABLE:

                    st.error(
                        "Face processing is unavailable."
                    )

                else:

                    safe_name = "".join(
                        c for c in name
                        if c.isalnum()
                        or c in (" ", "-", "_")
                    ).strip()

                    gray = image_to_gray(picture)

                    if gray is None:

                        st.error(
                            "Could not process the photo."
                        )

                    else:

                        face = get_face(gray)

                        if face is None:

                            st.error(
                                "Exactly one clear face must be visible. "
                                "Please retake the photo."
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

                            existing = [
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

                            number = len(existing) + 1

                            path = os.path.join(
                                student_folder,
                                f"{number}.jpg"
                            )

                            success = cv2.imwrite(
                                path,
                                face
                            )

                            if success:

                                st.success(
                                    f"✅ {safe_name} registered!"
                                )

                                st.session_state.register_mode = False

                                st.rerun()

                            else:

                                st.error(
                                    "Could not save the student's photo."
                                )

        with cancel_col:

            if st.button(
                "❌ Cancel",
                use_container_width=True
            ):

                st.session_state.register_mode = False
                st.rerun()

# --------------------------------------------------
# TAKE ATTENDANCE
# --------------------------------------------------

if st.session_state.attendance_mode:

    st.divider()

    st.subheader("🎥 Take Attendance")

    st.info(
        "Position one registered student clearly in front of the camera."
    )

    attendance_photo = st.camera_input(
        "📷 Capture Student",
        key="attendance_camera"
    )

    if attendance_photo is not None:

        st.image(
            attendance_photo,
            caption="Captured Attendance Photo",
            width=350
        )

        if st.button(
            "🔍 Recognize & Mark Attendance",
            type="primary",
            use_container_width=True
        ):

            if not OPENCV_AVAILABLE:

                st.error(
                    "Face recognition is unavailable on the server."
                )

            elif registered_students == 0:

                st.warning(
                    "Register at least one student first."
                )

            else:

                with st.spinner(
                    "Recognizing student..."
                ):

                    name, score, error = recognize_student(
                        attendance_photo
                    )

                if error:

                    st.error(
                        f"❌ {error}"
                    )

                else:

                    added = save_attendance(name)

                    if added:

                        st.success(
                            f"✅ {name} recognized. Attendance marked!"
                        )

                    else:

                        st.info(
                            f"ℹ️ {name} is already marked present today."
                        )

                    st.session_state.attendance_mode = False

                    st.rerun()

    if st.button(
        "❌ Close Attendance Camera"
    ):

        st.session_state.attendance_mode = False
        st.rerun()

st.divider()

# --------------------------------------------------
# TODAY'S ATTENDANCE
# --------------------------------------------------

st.subheader("📅 Today's Attendance")

attendance = load_attendance()

if not attendance.empty:

    today_attendance = attendance[
        attendance["Date"].astype(str) == today
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

st.divider()

# --------------------------------------------------
# ATTENDANCE HISTORY
# --------------------------------------------------

st.subheader("📚 Attendance History")

attendance = load_attendance()

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