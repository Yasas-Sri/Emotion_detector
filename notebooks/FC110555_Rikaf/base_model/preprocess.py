import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder


IMAGE_SIZE = tuple([48, 48])     
TRAIN_PATH = "./../../../Data/images/train"
VAL_PATH = "./../../../Data/images/validation"


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

def get_feature_datasets(train_csv_path, val_csv_path):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if 'label' not in train_df.columns or 'label' not in val_df.columns:
        raise ValueError("Both CSV files must contain a 'label' column.")

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    X_val = val_df.drop(columns=["label"]).values
    y_val = val_df["label"].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    return X_train, y_train, X_val, y_val, le

