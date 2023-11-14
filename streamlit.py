pip install tensorflow
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

# Load models
dn_model = tf.keras.models.load_model('C:/Users/lvminglin/Desktop/cv/assignment 3/model_tf.h5')
cnn_model = tf.keras.models.load_model('C:/Users/lvminglin/Desktop/cv/assignment 3/model_cnn.h5')

@st.cache
def predict_image(model, image):
    image = np.array(image.resize((64, 64))).astype('float32') / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

st.title('Image Predictor')

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Select the Model
    model_option = st.selectbox('Choose a model:', ('CNN Model', 'DenseNet Model'))

    # Use the selected model
    if model_option == 'CNN Model':
        selected_model = cnn_model
    else:
        selected_model = dn_model

    image = Image.open(uploaded_file)
    predictions = predict_image(selected_model, image)
    
    class_names = ['0', '1', '10', '100', '101', '11', '13', '14', '15', '16', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '37', '38', '39', '4', '40', '41', '42', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '54', '56', '57', '58', '59', '6', '62', '64', '66', '67', '68', '69', '7', '70', '71', '73', '74', '76', '77', '78', '79', '8', '82', '83', '84', '86', '87', '88', '9', '90', '91', '92', '93', '94', '95', '97', '99']
    st.write(f"Prediction: {class_names[np.argmax(predictions[0])]}")


