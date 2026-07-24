import cv2
import os

MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.txt"

# Check required files
if not os.path.exists(MODEL_FILE):
    print("ERROR: face_model.yml not found.")
    print("Run train_model.py first.")
    raise SystemExit

if not os.path.exists(LABELS_FILE):
    print("ERROR: labels.txt not found.")
    raise SystemExit

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# Load names
labels = {}

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        label, name = line.split(",", 1)
        labels[int(label)] = name

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("Face recognition started.")
print("Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Camera frame could not be read.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        # LBPH: lower confidence value means a closer match
        if confidence < 70 and label in labels:
            name = labels[label]
            text = f"{name} ({confidence:.0f})"
        else:
            name = "Unknown"
            text = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()