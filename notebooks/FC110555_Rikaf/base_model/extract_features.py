import os
import cv2
import mediapipe as mp
import numpy as np
import csv

IMAGE_FOLDER = "./../../../Data/images/images/validation"     
CSV_FILE = "validation_features.csv"
INCLUDE_LABEL = True         
IMAGE_SIZE = (224, 224)      

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    coords = [(p.x * w, p.y * h) for p in landmarks.landmark]
    return coords

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    coords = extract_landmarks(image)
    if coords is None:
        return None
    flat_coords = []
    for x, y in coords:
        flat_coords.extend([x, y])
    return flat_coords

def main():
    all_data = []

    num_landmarks = 468
    headers = [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(num_landmarks * 2)]
    if INCLUDE_LABEL:
        headers.append("label")

    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                features = process_image(image_path)
                if features:
                    if INCLUDE_LABEL:
                        label = os.path.basename(root)
                        features.append(label)
                    all_data.append(features)
                    print(f"Processed {image_path}")

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_data)

    print(f"Saved {len(all_data)} entries to {CSV_FILE}")

if __name__ == "__main__":
    main()
