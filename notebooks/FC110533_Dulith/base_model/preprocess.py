iimport os
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder


IMAGE_SIZE: Tuple[int, int] = (227, 227)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "Data" / "images" / "train"
DEFAULT_VAL_PATH = PROJECT_ROOT / "Data" / "images" / "validation"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "notebooks" / "FC110533_Dulith" / "base_model" / "preprocessed_data"


TRAIN_PATH = os.environ.get("EMOTION_TRAIN_PATH", str(DEFAULT_TRAIN_PATH))
VAL_PATH = os.environ.get("EMOTION_VAL_PATH", str(DEFAULT_VAL_PATH))
SAVE_DIR = os.environ.get("EMOTION_PREPROCESS_OUT", str(DEFAULT_SAVE_DIR))


USE_EQUALIZE = True         
SCALE_0_1    = True          
Z_SCORE      = True          
REPEAT_TO_RGB = True         


def _list_image_files(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    files.sort()
    return files

def _load_grayscale(path: str, size: Tuple[int, int]) -> np.ndarray:
    """
    Load an image, convert to grayscale, optional equalize, resize, return float32 HxW.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        if USE_EQUALIZE:
            im = ImageOps.equalize(im)
        im = im.resize(size, Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32)  
    return x

def _normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize a grayscale array HxW.
    - Optional scale to [0,1]
    - Optional per-image z-score (mean 0, std 1), with epsilon guard
    """
    if SCALE_0_1:
        x = x / 255.0
    if Z_SCORE:
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-6:
            sigma = 1e-6
        x = (x - mu) / sigma
    return x

def _to_channels(x_gray: np.ndarray) -> np.ndarray:
    """
    Convert HxW to HxWxC (C=3 if REPEAT_TO_RGB else 1)
    """
    if REPEAT_TO_RGB:
        return np.repeat(x_gray[..., None], 3, axis=-1)
    else:
        return x_gray[..., None]

def _compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Balanced class weights: n_samples / (n_classes * count_c)
    """
    counts = Counter(y.tolist())
    n = len(y)
    weights = {}
    for c in range(num_classes):
        count_c = counts.get(c, 1)
        weights[c] = n / (num_classes * count_c)
    return weights


def process_and_save_images(split_path: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

   
    class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    if not class_names:
        raise RuntimeError(f"No class folders found under: {split_path}")

   
    le = LabelEncoder()
    le.fit(class_names)

   
    per_class_paths: Dict[str, List[str]] = {}
    total_images = 0
    for cls in class_names:
        cls_dir = os.path.join(split_path, cls)
        img_paths = _list_image_files(cls_dir)
        per_class_paths[cls] = img_paths
        total_images += len(img_paths)

    if total_images == 0:
        raise RuntimeError(f"No images found under: {split_path}")

    num_channels = 3 if REPEAT_TO_RGB else 1
    img_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], num_channels)
    images_path = os.path.join(save_dir, "images.npy")
    X = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_images, *img_shape),
    )
    y = np.empty(total_images, dtype=np.int64)

    idx = 0
    skipped = 0
    for cls in class_names:
        cls_label = le.transform([cls])[0]
        for pth in per_class_paths[cls]:
            try:
                x = _load_grayscale(pth, IMAGE_SIZE)
                x = _normalize(x)
                x = _to_channels(x)
                X[idx] = x
                y[idx] = cls_label
                idx += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] Failed to process {pth}: {e}")

    if idx == 0:
        del X
        os.remove(images_path)
        raise RuntimeError(f"No images processed successfully under: {split_path}")

    if idx != total_images:
        print(f"[WARN] Processed {idx} / {total_images} images under {split_path}. Truncating output arrays.")
        tmp_path = images_path + ".tmp"
        truncated = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(idx, *img_shape),
        )
        block = 512
        for start in range(0, idx, block):
            end = min(idx, start + block)
            truncated[start:end] = X[start:end]
        truncated.flush()
        del truncated
        del X
        os.replace(tmp_path, images_path)
    else:
        X.flush()
        del X

    y = y[:idx]

   
    np.save(os.path.join(save_dir, "labels.npy"), y)

   
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

 
    meta = {
        "num_samples": int(idx),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "image_shape": list(img_shape),  
        "scale_0_1": SCALE_0_1,
        "z_score": Z_SCORE,
        "equalize": USE_EQUALIZE,
        "repeat_to_rgb": REPEAT_TO_RGB,
        "image_size": list(IMAGE_SIZE),
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    
    cw = _compute_class_weights(y, num_classes=len(class_names))
    with open(os.path.join(save_dir, "class_weights.pkl"), "wb") as f:
        pickle.dump(cw, f)

    print(f"[OK] {split_path} -> {save_dir}")
    print(f"     images: {(idx, *img_shape)}, labels: {y.shape}")
    print(f"     classes: {class_names}")
    print(f"     class weights: {cw}")
    if skipped:
        print(f"     skipped: {skipped} files (see warnings above)")

def save_global_label_encoder(train_split_dir: str, out_path: str) -> None:
    """
    Create a single label encoder based on the TRAIN split (canonical order),
    so your training/validation loaders decode consistently.
    """
    class_names = sorted([d for d in os.listdir(train_split_dir) if os.path.isdir(os.path.join(train_split_dir, d))])
    le = LabelEncoder()
    le.fit(class_names)
    with open(out_path, "wb") as f:
        pickle.dump(le, f)

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

   
    process_and_save_images(TRAIN_PATH, os.path.join(SAVE_DIR, "train"))
    process_and_save_images(VAL_PATH,   os.path.join(SAVE_DIR, "validation"))

   
    save_global_label_encoder(TRAIN_PATH, os.path.join(SAVE_DIR, "label_encoder.pkl"))

