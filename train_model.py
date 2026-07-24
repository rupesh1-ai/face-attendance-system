import cv2
import os
import numpy as np

DATASET_PATH = "faces"
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.txt"

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_images = []
face_labels = []
label_names = {}

current_label = 0

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_names[current_label] = person_name

    for filename in os.listdir(person_folder):

        image_path = os.path.join(person_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))

        face_images.append(image)
        face_labels.append(current_label)

    current_label += 1

if len(face_images) == 0:
    print("ERROR: No registered face images found.")
    exit()

recognizer.train(
    face_images,
    np.array(face_labels)
)

recognizer.save(MODEL_FILE)

with open(LABELS_FILE, "w", encoding="utf-8") as file:
    for label, name in label_names.items():
        file.write(f"{label},{name}\n")

print("Training completed successfully!")
print(f"People trained: {len(label_names)}")
print(f"Images trained: {len(face_images)}")
print(f"Model saved as: {MODEL_FILE}")