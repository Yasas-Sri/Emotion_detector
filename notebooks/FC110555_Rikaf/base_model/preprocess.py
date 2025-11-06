import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


IMAGE_SIZE = tuple([48, 48])     
TRAIN_PATH = "./../../../Data/images/train"
VAL_PATH = "./../../../Data/images/validation"

import numpy as np

def engineer_features(landmarks_flat):
    landmarks = landmarks_flat.reshape(-1, 2)

    nose_tip = landmarks[1]
    centered_landmarks = landmarks - nose_tip
    
    left_eye_corner = landmarks[33] 
    right_eye_corner = landmarks[263]
    scale = np.linalg.norm(right_eye_corner - left_eye_corner)
    if scale == 0:
        scale = 1
    normalized_landmarks = centered_landmarks / scale

    mouth_outer_indices = [61, 84, 17, 314, 405, 291, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    left_eyebrow_indices = [70, 63, 105, 66, 107]
    right_eyebrow_indices = [296, 334, 293, 300, 276]
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    key_points = np.concatenate([
        normalized_landmarks[mouth_outer_indices],
        normalized_landmarks[left_eyebrow_indices],
        normalized_landmarks[right_eyebrow_indices],
        normalized_landmarks[left_eye_indices],
        normalized_landmarks[right_eye_indices]
    ])

    distances = []
    for i in range(len(key_points)):
        for j in range(i + 1, len(key_points)):
            dist = np.linalg.norm(key_points[i] - key_points[j])
            distances.append(dist)
    
    left_eye_center = np.mean(normalized_landmarks[left_eye_indices], axis=0)
    right_eye_center = np.mean(normalized_landmarks[right_eye_indices], axis=0)
    left_eyebrow_center = np.mean(normalized_landmarks[left_eyebrow_indices], axis=0)
    right_eyebrow_center = np.mean(normalized_landmarks[right_eyebrow_indices], axis=0)
    
    eye_asymmetry_y = left_eye_center[1] - right_eye_center[1]
    eyebrow_asymmetry_y = left_eyebrow_center[1] - right_eyebrow_center[1]

    final_features = distances + [eye_asymmetry_y, eyebrow_asymmetry_y]
    
    return np.array(final_features)

def load_data_from_folder(base_path):
    X = []
    y = []
    class_names = os.listdir(base_path)
    
    for label in class_names:
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path): continue
        
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img).flatten()
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")
    
    return np.array(X), np.array(y)

def get_datasets():
    print("Loading training data...")
    X_train, y_train = load_data_from_folder(TRAIN_PATH)

    print("Loading validation data...")
    X_val, y_val = load_data_from_folder(VAL_PATH)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    return X_train, y_train, X_val, y_val, le

def normalize_landmarks(landmarks_flat):
    landmarks = landmarks_flat.reshape(-1, 2)
    
    nose_tip = landmarks[1]
    centered = landmarks - nose_tip
    
    left_eye_corner = landmarks[33]
    right_eye_corner = landmarks[263]
    eye_distance = np.linalg.norm(right_eye_corner - left_eye_corner)
    
    if eye_distance == 0:
        return np.zeros_like(landmarks_flat)
        
    normalized = centered / eye_distance
    
    return normalized.flatten()


def get_feature_datasets(train_csv_path, val_csv_path):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if 'label' not in train_df.columns or 'label' not in val_df.columns:
        raise ValueError("Both CSV files must contain a 'label' column.")

    train_landmarks = train_df.drop(columns=["label"]).values
    val_landmarks = val_df.drop(columns=["label"]).values
    
    print("Engineering features from landmarks...")
    X_train = np.array([engineer_features(x) for x in train_landmarks])
    X_val = np.array([engineer_features(x) for x in val_landmarks])
    print(f"New feature shape: {X_train.shape}")

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    return X_train, y_train, X_val, y_val, le