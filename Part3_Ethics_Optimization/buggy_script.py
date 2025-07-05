# BUGGY TENSORFLOW SCRIPT
# This script is intentionally buggy for the troubleshooting challenge.

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Load and Preprocess the Data ---

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Bug 1: Data is not reshaped for CNN.
# A CNN in TensorFlow/Keras expects a 4D tensor (batch_size, height, width, channels).
# The data is currently (60000, 28, 28), which will cause a dimension mismatch error.
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) # This line is missing
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)   # This line is missing

# Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode the labels
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

# --- 2. Build the CNN Model ---

model = Sequential([
    # The input shape is incorrect because the data was not reshaped.
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28)), # Should be (28, 28, 1)
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    # Bug 2: Incorrect activation function for the final layer.
    # For multi-class classification, 'softmax' is needed to output probabilities for each class.
    # 'relu' will output values from 0 to infinity, which is not suitable for classification.
    Dense(10, activation='relu')
])

# Bug 3: Incorrect loss function.
# 'binary_crossentropy' is for binary (2-class) classification.
# For multi-class classification (10 digits), 'categorical_crossentropy' should be used.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---

print("Attempting to train the buggy model...")
# This will likely fail with an error about input dimensions.
try:
    model.fit(X_train, y_train_categorical,
              batch_size=128,
              epochs=5,
              verbose=1,
              validation_data=(X_test, y_test_categorical))
except Exception as e:
    print(f"\n--- !!! An error occurred as expected !!! ---")
    print(f"Error: {e}")
    print("\nThis script has intentional bugs. Please debug and fix it.") 