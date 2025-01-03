
import tensorflow as tf
import os

# Check if model file exists
model_path = 'spy_stock_model.tflite'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please ensure the model file is present in the workspace.")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
