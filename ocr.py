# Multiclass Classification using Deep Neural Network
# OCR Letter Recognition Dataset

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==============================
# Load Dataset
# ==============================

# Download dataset from:
# https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

# Put the file in same folder as notebook

# Column names
columns = [
    'letter',
    'x-box', 'y-box', 'width', 'height',
    'onpix', 'x-bar', 'y-bar', 'x2bar',
    'y2bar', 'xybar', 'x2ybr', 'xy2br',
    'x-ege', 'xegvy', 'y-ege', 'yegvx'
]

# Read dataset
data = pd.read_csv("letter-recognition.data", names=columns)

# Display first 5 rows
print(data.head())

# ==============================
# Prepare Features and Labels
# ==============================

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode labels (A-Z -> 0-25)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-hot encoding
y_categorical = to_categorical(y_encoded)

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Build Deep Neural Network
# ==============================

model = Sequential()

# Input + Hidden Layers
model.add(Dense(128, activation='relu', input_shape=(16,)))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

# Output Layer (26 classes)
model.add(Dense(26, activation='softmax'))

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Model Summary
# ==============================

model.summary()

# ==============================
# Train Model
# ==============================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==============================
# Evaluate Model
# ==============================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)

# Convert predictions
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# ==============================
# Accuracy and Report
# ==============================

print("\nAccuracy Score:")
print(accuracy_score(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_true, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# Plot Accuracy Graph
# ==============================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()