import cv2
import os
import sys

# ---------------------------------
# Get student name from dashboard
# ---------------------------------

if len(sys.argv) < 2:
    print("Student name was not provided.")
    sys.exit()

name = sys.argv[1].strip()

if not name:
    print("Name cannot be empty.")
    sys.exit()

# ---------------------------------
# Create student's face folder
# ---------------------------------

folder = os.path.join("faces", name)
os.makedirs(folder, exist_ok=True)

# Remove old images if re-registering
for file in os.listdir(folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            os.remove(os.path.join(folder, file))
        except:
            pass

# ---------------------------------
# Load face detector
# ---------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Face detector could not be loaded.")
    sys.exit()

# ---------------------------------
# Open camera
# ---------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    sys.exit()

count = 0
MAX_IMAGES = 30

print(f"Registering: {name}")
print("Look at the camera.")
print("Press Q to cancel.")

# ---------------------------------
# Capture faces
# ---------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Camera could not be read.")
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

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (200, 200)
        )

        count += 1

        filename = os.path.join(
            folder,
            f"{count}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        cv2.putText(
            frame,
            f"Captured: {count}/{MAX_IMAGES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Prevent multiple detected faces
        # from being saved in one frame
        break

    # Student name
    cv2.putText(
        frame,
        name,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Instructions
    cv2.putText(
        frame,
        "Look at camera - Press Q to cancel",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Register Student",
        frame
    )

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break

    if count >= MAX_IMAGES:
        break

# ---------------------------------
# Cleanup
# ---------------------------------

camera.release()
cv2.destroyAllWindows()

# ---------------------------------
# Result
# ---------------------------------

if count >= MAX_IMAGES:

    print(
        f"Registration complete for {name}."
    )

    print(
        f"{count} face images saved."
    )

else:

    print(
        f"Registration stopped. "
        f"{count}/{MAX_IMAGES} images captured."
    )