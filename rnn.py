# Recurrent Neural Network (RNN)
# Google Stock Price Prediction using LSTM

# ==========================================
# Import Libraries
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ==========================================
# Load Dataset
# ==========================================

# Download dataset:
# https://www.kaggle.com/datasets/varpit94/google-stock-data

# Use Google_Stock_Price_Train.csv
# Put CSV file in same folder as notebook

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

print(dataset_train.head())

# ==========================================
# Select Open Price Column
# ==========================================

training_set = dataset_train.iloc[:, 1:2].values

# ==========================================
# Feature Scaling
# ==========================================

scaler = MinMaxScaler(feature_range=(0,1))

training_set_scaled = scaler.fit_transform(training_set)

# ==========================================
# Create Data Structure
# Using previous 60 days to predict next day
# ==========================================

X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for RNN
X_train = np.reshape(
    X_train,
    (X_train.shape[0], X_train.shape[1], 1)
)

print("X_train Shape :", X_train.shape)

# ==========================================
# Build RNN Model
# ==========================================

model = Sequential()

# First LSTM Layer
model.add(LSTM(
    units=50,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1)
))

model.add(Dropout(0.2))

# Second LSTM Layer
model.add(LSTM(
    units=50,
    return_sequences=True
))

model.add(Dropout(0.2))

# Third LSTM Layer
model.add(LSTM(
    units=50,
    return_sequences=True
))

model.add(Dropout(0.2))

# Fourth LSTM Layer
model.add(LSTM(
    units=50
))

model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units=1))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
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
    y_train,
    epochs=20,
    batch_size=32
)

# ==========================================
# Load Test Dataset
# ==========================================

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")

real_stock_price = dataset_test.iloc[:, 1:2].values

# ==========================================
# Prepare Test Data
# ==========================================

dataset_total = pd.concat(
    (dataset_train['Open'], dataset_test['Open']),
    axis=0
)

inputs = dataset_total[
    len(dataset_total) - len(dataset_test) - 60:
].values

inputs = inputs.reshape(-1,1)

inputs = scaler.transform(inputs)

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(
    X_test,
    (X_test.shape[0], X_test.shape[1], 1)
)

# ==========================================
# Predict Stock Prices
# ==========================================

predicted_stock_price = model.predict(X_test)

# Reverse Scaling
predicted_stock_price = scaler.inverse_transform(
    predicted_stock_price
)

# ==========================================
# Visualize Results
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    real_stock_price,
    color='red',
    label='Real Google Stock Price'
)

plt.plot(
    predicted_stock_price,
    color='blue',
    label='Predicted Google Stock Price'
)

plt.title('Google Stock Price Prediction using RNN')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')

plt.legend()

plt.show()

# ==========================================
# Plot Training Loss
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()