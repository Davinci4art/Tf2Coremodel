
import numpy as np
import pandas as pd
import tensorflow as tf
import coremltools as ct
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import os

# Load SPY ETF data
data = yf.download('SPY', start='2010-01-01', end='2024-12-31')

# Preprocess the data
data['Close'] = data['Close'].fillna(method='ffill')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define and train LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Save as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("spy_stock_model.tflite", "wb") as f:
    f.write(tflite_model)

# Load TFLite model for CoreML conversion
interpreter = tf.lite.Interpreter(model_path='spy_stock_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Convert to CoreML
input_shape = tuple(input_details[0]['shape'])
output_shape = tuple(output_details[0]['shape'])

mlmodel = ct.convert(
    'spy_stock_model.tflite',
    source='tensorflow',
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=input_shape, name="input_1")],
    outputs=[ct.TensorType(shape=output_shape, name="output_1")],
    minimum_deployment_target=ct.target.iOS13
)

# Save the Core ML model
mlmodel.save('StockPatternClassifier.mlmodel')
print("Model converted and saved as StockPatternClassifier.mlmodel")
