
import tensorflow as tf
import coremltools as ct
import os

# Check if model file exists
model_path = 'stock_pattern_cnn.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please ensure the model file is present in the workspace.")

# Load the TensorFlow model
tf_model = tf.keras.models.load_model(model_path)

# Convert to Core ML
ml_model = ct.convert(tf_model, source='tensorflow')

# Save the Core ML model
ml_model.save('StockPatternClassifier.mlmodel')
