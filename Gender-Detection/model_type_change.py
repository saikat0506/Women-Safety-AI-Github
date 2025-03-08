from tensorflow.keras.models import load_model

# Load the old model
old_model = load_model('gender_detection.model')

# Save it in the new format
old_model.save('gender_detection.keras')  # Save in Keras 3 format
old_model.save('gender_detection.h5')  # Save in HDF5 format
