import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

IMAGE_SIZE = (128, 128)
TRAIN_PATH = "/workspaces/Emotion_detector/Data/images/images/train"
VAL_PATH = "/workspaces/Emotion_detector/Data/images/images/validation"
SAVE_DIR = "/workspaces/Emotion_detector/notebooks/FC110533_Dulith/base_model/preprocessed_data"
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def process_and_save_images(base_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    le = LabelEncoder()
    le.fit(CLASS_NAMES)

    for label in CLASS_NAMES:
        class_dir = os.path.join(base_path, label)
        save_class_dir = os.path.join(save_dir, label)
        os.makedirs(save_class_dir, exist_ok=True)
        idx = 0
        for img_name in os.listdir(class_dir):
            try:
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert('L').resize(IMAGE_SIZE)
                img_arr = np.array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=-1)  # shape: (128, 128, 1)
                # Save each image as a separate npy file in its class folder
                np.save(os.path.join(save_class_dir, f"img_{idx}.npy"), img_arr)
                idx += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print(f"Processed and saved {idx} images in {save_class_dir}")

    # Save the label encoder
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

if __name__ == "__main__":
    process_and_save_images(TRAIN_PATH, os.path.join(SAVE_DIR, "train"))
    process_and_save_images(VAL_PATH, os.path.join(SAVE_DIR, "validation"))
