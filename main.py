
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

# Load SPY ETF data with error handling
try:
    data = yf.download('SPY', start='2010-01-01', end='2024-12-31')
    if data.empty:
        raise ValueError("No data downloaded")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit(1)

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
X_train = X_train.reshape((X_train.shape[0], sequence_length, 1))
X_test = X_test.reshape((X_test.shape[0], sequence_length, 1))

# Define and train LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model and validate predictions
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

# Make predictions to validate model
try:
    test_predictions = model.predict(X_test[:1])
    if np.isnan(test_predictions).any():
        raise ValueError("Model predictions contain NaN values")
    print("Model predictions validated successfully")

    # Save complete model with memory optimization
    tf.keras.backend.clear_session()
    model.save('spy_stock_model.keras', save_format='keras_v3')
    
    # Verify the saved model
    loaded_model = tf.keras.models.load_model('spy_stock_model.keras')
    if not loaded_model:
        raise ValueError("Failed to verify saved model")
    
    # Convert to CoreML with enhanced error handling and optimization
    spec = ct.convert(
        'spy_stock_model.keras',
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        source="tensorflow",
        inputs=[ct.TensorType(name="input_1", shape=(1, sequence_length, 1))],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=False
    )
    
    # Save the CoreML model
    spec.save("StockPatternClassifier.mlmodel")
    print("Model converted and saved successfully as StockPatternClassifier.mlmodel")
    
except Exception as e:
    print(f"Error during model saving/conversion: {e}")
    exit(1)
