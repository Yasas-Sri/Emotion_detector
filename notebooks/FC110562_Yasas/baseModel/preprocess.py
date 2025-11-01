
from keras import layers, models
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.preprocessing import image_dataset_from_directory



def get_augmentation_model(input_shape=(48, 48, 1)):
    return models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Rescaling(1./255.0),  # Normalize pixel values to 0–1
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.0833),  
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  
    ])



def get_data_splits(img_size=(48, 48), batch_size=32, val_split=0.5):
    train_ds = image_dataset_from_directory(
        "../../Data/images/train",
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Load validation data  first
    full_val_ds = image_dataset_from_directory(
        "../../Data/images/validation",
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=img_size,
        batch_size=None,  
        shuffle=False,
        seed=42
    )
    
    # Get the total number of samples
    val_size = tf.data.experimental.cardinality(full_val_ds).numpy()
    split_idx = int(val_size * val_split)
    
    # Split the dataset
    val_ds = full_val_ds.take(split_idx)
    test_ds = full_val_ds.skip(split_idx)
    
    # Now batch the split datasets
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    return train_ds, val_ds, test_ds


