
import tensorflow as tf
import coremltools as ct
import os

# Check if model file exists
model_path = 'spy_stock_model.tflite'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please ensure the model file is present in the workspace.")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load TF model from TFLite
converter = tf.lite.TFLiteConverter.from_file(model_path)
tf_model = converter.convert()

# Convert to Core ML
mlmodel = ct.convert(
    tf_model,
    source='tensorflow',
    minimum_deployment_target=ct.target.iOS13
)

# Save the Core ML model
mlmodel.save('StockPatternClassifier.mlmodel')
print("Model converted and saved as StockPatternClassifier.mlmodel")
