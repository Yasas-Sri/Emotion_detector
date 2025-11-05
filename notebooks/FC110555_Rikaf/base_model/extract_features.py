import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import argparse
from sklearn.model_selection import train_test_split


NUM_LANDMARKS = 468
LANDMARKS_PER_IMAGE = NUM_LANDMARKS * 2

def extract_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    coords = [(p.x, p.y) for p in landmarks.landmark]
    return coords

def process_and_write_csv(image_paths, labels, output_csv, include_label):
    if not image_paths:
        print(f"No images to process for {output_csv}. Skipping.")
        return 0, 0

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        
        processed_count = 0
        skipped_count = 0
        
        headers = [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(LANDMARKS_PER_IMAGE)]
        if include_label:
            headers.append("label")

        with open(output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for image_path, label in zip(image_paths, labels):
                image = cv2.imread(image_path)
                if image is None:
                    skipped_count += 1
                    continue
                
                coords = extract_landmarks(image, face_mesh)
                if coords is None:
                    skipped_count += 1
                    continue
                
                flat_coords = [coord for point in coords for coord in point]
                
                if include_label:
                    flat_coords.append(label)
                
                writer.writerow(flat_coords)
                processed_count += 1
                print(f"Processing {os.path.basename(output_csv)}: {processed_count} images", end='\r')

        print(f"\nFinished {os.path.basename(output_csv)}: {processed_count} processed, {skipped_count} skipped.")
        return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description='Extract and split facial landmarks into train, validation, and test CSVs.')

    parser.add_argument('--source-folder', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--include-label', action='store_true')


    args = parser.parse_args()

    if not os.path.isdir(args.source_folder):
        print(f"Error: Source folder not found at {args.source_folder}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- Step 1: Collecting all image paths and labels ---")
    all_image_paths = []
    all_labels = []
    for label in os.listdir(args.source_folder):
        label_path = os.path.join(args.source_folder, label)
        if not os.path.isdir(label_path): continue
        
        for img_name in os.listdir(label_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(label_path, img_name))
                all_labels.append(label)
    
    print(f"Found {len(all_image_paths)} images across {len(set(all_labels))} classes.")

    print("\n--- Step 2: Splitting data into Train, validation, and Test sets ---")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_image_paths, all_labels, 
        test_size=args.test_size, 
        stratify=all_labels, 
        random_state=42 
    )

    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=val_size_adjusted, 
        stratify=y_trainval,
        random_state=42
    )
    
    print(f"data split complete:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Test: {len(X_test)} images")

    print("\n--- Step 3: Extracting features and writing to CSVs ---")
    train_csv_path = os.path.join(args.output_dir, "train_features.csv")
    val_csv_path = os.path.join(args.output_dir, "validation_features.csv")
    test_csv_path = os.path.join(args.output_dir, "test_features.csv")

    process_and_write_csv(X_train, y_train, train_csv_path, args.include_label)
    
    process_and_write_csv(X_val, y_val, val_csv_path, args.include_label)

    process_and_write_csv(X_test, y_test, test_csv_path, args.include_label)

    print("\n--- Extraction and Splitting Summary --")
    print(f"Training data:   -> {train_csv_path}")
    print(f"Validation data: -> {val_csv_path}")
    print(f"Test data:       -> {test_csv_path}")
    print("------------------")


if __name__ == "__main__":
    main()

# python extract_features.py --source-folder ./../../../Data/images/images/train --output-dir ./ --include-label --val-size 0.15 --test-size 0.15