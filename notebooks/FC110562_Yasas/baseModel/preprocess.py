
from keras import layers, models
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.preprocessing import image_dataset_from_directory
import os
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_augmentation_model(input_shape=(48, 48, 1)):
    return models.Sequential([
        #layers.InputLayer(input_shape=(48, 48, 3)),  # Accepts RGB input from dataset
        # layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),  # Converts to (48, 48, 1) 
        layers.InputLayer(input_shape=input_shape),
        layers.Rescaling(1./255.0),  # Normalize pixel values to 0–1
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.0833),  
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  
    ])


# def get_data_splits(img_size=(48, 48),   batch_size=64,val_split=0.1,seed=42,train_dir="../../../Data/images/train"):

#      train_ds = image_dataset_from_directory(
#         train_dir,
#         validation_split=val_split,
       
#         color_mode="grayscale",
#         subset="training",
#         seed=seed,
#         image_size=img_size,
#         batch_size=batch_size,
#         label_mode="categorical"  
#     )

#      val_ds = image_dataset_from_directory(
#         train_dir,
#         validation_split=val_split,
#           label_mode="categorical",
#         color_mode="grayscale",
#         subset="validation",
#         seed=seed,
#         image_size=img_size,
#         batch_size=batch_size,
#     )
     
#      def normalize(image, label):
#         return tf.cast(image, tf.float32) / 255.0, label

#      train_ds = train_ds.map(normalize)
#      val_ds = val_ds.map(normalize)

#      return train_ds, val_ds


def get_data_splits(img_size=(48, 48), batch_size=64, val_split=0.1, seed=42,
                    train_dir="../../../Data/images/train",
                    test_dir="../../../Data/images/validation"):
    # Split only inside the "train" folder
    train_ds = image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical"
    )

    val_ds = image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical"
    )

    # Test dataset → taken directly from the validation folder
    test_ds = image_dataset_from_directory(
        test_dir,
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical"
    )

    # Normalization
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_ds = train_ds.map(normalize)
    val_ds = val_ds.map(normalize)
    test_ds = test_ds.map(normalize)

    return train_ds, val_ds, test_ds




# def get_augmented_data_generators(folder_path, picture_size=48, batch_size=64):

#     datagen_train = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=10,
#         zoom_range=0.1,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=True
#     )

#     datagen_val = ImageDataGenerator(rescale=1./255)

#     train_ds = datagen_train.flow_from_directory(
#         folder_path + "train",
#         target_size=(picture_size, picture_size),
#         color_mode="grayscale",
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=True
#     )

#     val_ds = datagen_val.flow_from_directory(
#         folder_path + "validation",
#         target_size=(picture_size, picture_size),
#         color_mode="grayscale",
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=False
#     )

#     return train_ds, val_ds



def get_augmented_data_generators(folder_path, picture_size=48, batch_size=64, val_split=0.1, seed=42):
    # Train generator with split
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Train & validation from train folder
    train_ds = datagen.flow_from_directory(
        folder_path + "train",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset="training",
        seed=seed
    )

    val_ds = datagen.flow_from_directory(
        folder_path + "train",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset="validation",
        seed=seed
    )

    # Test data from "validation" folder
    datagen_test = ImageDataGenerator(rescale=1./255)
    test_ds = datagen_test.flow_from_directory(
        folder_path + "validation",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_ds, val_ds, test_ds





def get_train_val_data_gen(folder_path, picture_size=48, batch_size=64, val_split=0.1, seed=42):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_ds = datagen.flow_from_directory(
        folder_path + "train",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset="training",
        seed=seed
    )

    val_ds = datagen.flow_from_directory(
        folder_path + "train",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset="validation",
        seed=seed
    )

    return train_ds, val_ds


def get_test_data_gen(folder_path, picture_size=48, batch_size=64):
    datagen_test = ImageDataGenerator(rescale=1./255)
    test_ds = datagen_test.flow_from_directory(
        folder_path + "validation",
        target_size=(picture_size, picture_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return test_ds
