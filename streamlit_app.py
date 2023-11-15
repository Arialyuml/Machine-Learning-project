import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Load the DenseNet model when the application starts
dn_model = load_model('model_tf.h5')

# Define class labels
class_labels = ['0', '1', '10', '100', '101', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

st.title("Image Classifier")


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Load and convert the image to grayscale
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Reshape from (150, 150) to (150, 150, 1)

        pred = dn_model.predict(img)  # Make prediction
        st.write(f"Prediction: {class_labels[np.argmax(pred[0])]}")
