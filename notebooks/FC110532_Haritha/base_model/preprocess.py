import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

# Configuration for Mini Xception
IMAGE_SIZE = (64, 64)
TRAIN_PATH = "/workspaces/Emotion_detector/Data/images/train"
VAL_PATH = "/workspaces/Emotion_detector/Data/images/validation"
SAVE_DIR = "/workspaces/Emotion_detector/notebooks/FC110532_Haritha/base_model/preprocessed_data"

def process_and_save_images(base_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = sorted([d for d in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, d))])
    
    le = LabelEncoder()
    le.fit(class_names)
    
    labels_list = []
    idx = 0
    
    for label in class_names:
        class_dir = os.path.join(base_path, label)
        for img_name in os.listdir(class_dir):
            try:
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert('L').resize(IMAGE_SIZE)
                img_arr = np.array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=-1)  # shape: (64, 64, 1)
                
                # Save each image as a separate npy file
                np.save(os.path.join(save_dir, f"img_{idx}.npy"), img_arr)
                labels_list.append(label)
                idx += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Encode and save labels
    labels_encoded = le.transform(labels_list)
    np.save(os.path.join(save_dir, "labels.npy"), labels_encoded)
    
    # Save the label encoder
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    print(f"Processed and saved {idx} images in {save_dir}")

if __name__ == "__main__":
    process_and_save_images(TRAIN_PATH, os.path.join(SAVE_DIR, "train"))
    process_and_save_images(VAL_PATH, os.path.join(SAVE_DIR, "validation"))