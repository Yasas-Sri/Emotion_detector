import os
import numpy as np
import pickle
from model import create_mini_xception, compile_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Paths
TRAIN_DIR = "/workspaces/Emotion_detector/notebooks/FC110532_Haritha/base_model/preprocessed_data/train"
VAL_DIR = "/workspaces/Emotion_detector/notebooks/FC110532_Haritha/base_model/preprocessed_data/validation"
MODEL_SAVE_PATH = "/workspaces/Emotion_detector/notebooks/FC110532_Haritha/base_model/mini_xception_model.h5"

def load_data(data_dir):
    """
    Load preprocessed images and labels
    """
    print(f"Loading data from {data_dir}...")
    
    # Load labels
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    
    # Load images
    images = []
    for i in range(len(labels)):
        img_path = os.path.join(data_dir, f"img_{i}.npy")
        img = np.load(img_path)
        images.append(img)
    
    images = np.array(images)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    return images, labels

def train_model():
    """
    Train the Mini Xception model
    """
    print("Starting Mini Xception training...")
    
    # Load data
    train_images, train_labels = load_data(TRAIN_DIR)
    val_images, val_labels = load_data(VAL_DIR)
    
    # Load label encoder to get number of classes
    with open(os.path.join(TRAIN_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")
    
    # Create and compile model
    model = create_mini_xception(num_classes=num_classes)
    model = compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
    val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()