import cv2
import os
import csv
from datetime import datetime

MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.txt"
ATTENDANCE_FILE = "attendance.csv"

# -------------------------
# Check required files
# -------------------------

if not os.path.exists(MODEL_FILE):
    print("ERROR: face_model.yml not found.")
    raise SystemExit

if not os.path.exists(LABELS_FILE):
    print("ERROR: labels.txt not found.")
    raise SystemExit


# -------------------------
# Load recognition model
# -------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)


# -------------------------
# Load registered names
# -------------------------

labels = {}

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if line:
            label, name = line.split(",", 1)
            labels[int(label)] = name


# -------------------------
# Face detector
# -------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# -------------------------
# Create CSV if needed
# -------------------------

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time",
            "Status"
        ])


# -------------------------
# Load today's attendance
# -------------------------

today = datetime.now().strftime("%d-%m-%Y")

marked_today = set()

with open(
    ATTENDANCE_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["Date"] == today:
            marked_today.add(row["Name"])


print("Already marked today:", marked_today)


# -------------------------
# Attendance function
# -------------------------

def mark_attendance(name):

    if name in marked_today:

        print(
            f"{name} is already marked present today."
        )

        return False

    now = datetime.now()

    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            date,
            time,
            "Present"
        ])

    marked_today.add(name)

    print(
        f"Attendance marked: "
        f"{name} | {date} | {time}"
    )

    return True


# -------------------------
# Open camera
# -------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    raise SystemExit


print("")
print("================================")
print("   FACE ATTENDANCE SYSTEM")
print("================================")
print("")
print("Camera started.")
print("Press Q to exit.")
print("")


# -------------------------
# Main loop
# -------------------------

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Camera frame could not be read.")

        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    for (x, y, w, h) in faces:

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (200, 200)
        )


        label, confidence = recognizer.predict(face)


        # -------------------------
        # Registered person
        # -------------------------

        if confidence < 70 and label in labels:

            name = labels[label]

            if name in marked_today:

                display_text = (
                    f"{name} - Already Present"
                )

            else:

                mark_attendance(name)

                display_text = (
                    f"{name} - Attendance Marked"
                )


            color = (0, 255, 0)


        # -------------------------
        # Unknown person
        # -------------------------

        else:

            display_text = "Unknown"

            color = (0, 0, 255)


        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )


        # Name/status
        cv2.putText(
            frame,
            display_text,
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    cv2.imshow(
        "Face Attendance System",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

camera.release()

cv2.destroyAllWindows()

print("")
print("Attendance system stopped.")
