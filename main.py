import tensorflow as tf
import coremltools as ct

# Load the TensorFlow model
tf_model = tf.keras.models.load_model('stock_pattern_cnn.h5')

# Convert to Core ML
ml_model = ct.convert(tf_model, source='tensorflow')

# Save the Core ML model
ml_model.save('StockPatternClassifier.mlmodel')