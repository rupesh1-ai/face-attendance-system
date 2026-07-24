import cv2
import time

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Try DirectShow first
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Give camera time to start
time.sleep(2)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)
    time.sleep(2)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

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
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
