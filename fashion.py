# CNN for Fashion MNIST Classification
# Deep Learning using Convolutional Neural Network (CNN)

# ==========================================
# Import Libraries
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# ==========================================
# Load Fashion MNIST Dataset
# ==========================================

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# ==========================================
# Dataset Information
# ==========================================

print("Training Images Shape :", X_train.shape)
print("Testing Images Shape :", X_test.shape)

# ==========================================
# Class Labels
# ==========================================

fashion_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ==========================================
# Display Sample Images
# ==========================================

plt.figure(figsize=(10,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(fashion_classes[y_train[i]])
    plt.axis('off')

plt.show()

# ==========================================
# Normalize Data
# ==========================================

X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# Reshape Data for CNN
# ==========================================

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ==========================================
# One-Hot Encoding
# ==========================================

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ==========================================
# Build CNN Model
# ==========================================

model = Sequential()

# First Convolution Layer
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=(28,28,1)
))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu'
))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(10, activation='softmax'))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test_cat)

print("\nTest Accuracy :", accuracy)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred_classes,
    target_names=fashion_classes
))

# ==========================================
# Plot Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()

# ==========================================
# Plot Loss Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()

# ==========================================
# Display Predictions
# ==========================================

plt.figure(figsize=(12,12))

for i in range(9):
    plt.subplot(3,3,i+1)

    plt.imshow(X_test[i].reshape(28,28), cmap='gray')

    actual = fashion_classes[y_test[i]]
    predicted = fashion_classes[y_pred_classes[i]]

    plt.title(f"Actual: {actual}\nPred: {predicted}")

    plt.axis('off')

plt.tight_layout()
plt.show()