import tensorflow as tf
from tensorflow.keras import layers, models

def create_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    """
    Mini Xception model for emotion detection
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Block 2 - Depthwise separable
        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3 - Depthwise separable
        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(128, (1, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4 - Depthwise separable
        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(256, (1, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Global pooling and classification
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):
    """
    Compile the model
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Create and compile model
    model = create_mini_xception()
    model = compile_model(model)
    
    # Print summary
    model.summary()
    print(f"Total parameters: {model.count_params():,}")