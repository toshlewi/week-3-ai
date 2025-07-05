# FIXED TENSORFLOW SCRIPT
# This script contains the fixes for the buggy version.

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Load and Preprocess the Data ---

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# FIX 1: Reshape the data to be 4D (add channel dimension)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode the labels
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

print(f"Training data shape (fixed): {X_train.shape}")
print("-" * 30)

# --- 2. Build the CNN Model ---

model = Sequential([
    # FIX 1 (cont.): Update the input shape for the first layer
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    # FIX 2: Use 'softmax' for multi-class classification
    Dense(10, activation='softmax')
])

# FIX 3: Use 'categorical_crossentropy' for multi-class classification
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("-" * 30)

# --- 3. Train the Model ---

print("Training the fixed model...")
history = model.fit(X_train, y_train_categorical,
                    batch_size=128,
                    epochs=5,
                    verbose=1,
                    validation_data=(X_test, y_test_categorical))
print("Model training complete.")
print("-" * 30)

# --- 4. Evaluate the Model ---
score = model.evaluate(X_test, y_test_categorical, verbose=0)
print('--- Model Evaluation ---')
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')
print("-" * 30) 