import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the DenseNet model (model_tf.h5)
dn_model = tf.keras.models.load_model('model_tf.h5')

@st.cache
def predict_image(model, image):
    # Resize the image
    image = image.resize((64, 64))

    # Convert to grayscale if model expects 1 channel input
    if model.input_shape[-1] == 1:
        image = image.convert("L")

    # Convert image to numpy array and normalize
    image = np.array(image).astype('float32') / 255.0

    # Expand dimensions to match input shape
    if len(image.shape) == 2:  # For grayscale images
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    return predictions

st.title('Image Predictor')

uploaded_file = st.file_uploader("Choose an image...")  # File uploader widget

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    image = Image.open(uploaded_file)
    predictions = predict_image(dn_model, image)

        # Check if predictions are empty or not
    if predictions.size > 0:
        class_names = ['0', '1', '10', '100', '101', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81']
        predicted_class = class_names[np.argmax(predictions[0])]
        st.write(f"Prediction: {predicted_class}")
    else:
        st.write("Unable to make a prediction.")


