# Boston Housing Price Prediction using Deep Neural Network
# Jupyter Notebook Code

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load Dataset
# NOTE:
# load_boston() may be removed in latest sklearn versions.
# If error occurs, use:
# pip install scikit-learn==1.1.3

boston = load_boston()

# Features and Target
X = boston.data
y = boston.target

# Display Dataset Information
print("Feature Shape :", X.shape)
print("Target Shape :", y.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network Model
model = Sequential()

# Input + Hidden Layers
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output Layer
model.add(Dense(1))

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Model Summary
model.summary()

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Predict on Test Data
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("RMSE :", rmse)
print("R2 Score :", r2)

# Plot Training and Validation Loss
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()

# Compare Actual vs Predicted
comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred.flatten()
})

print(comparison.head(10))