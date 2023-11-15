import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the DenseNet model when the application starts
dn_model = load_model('model_tf.h5')

# Define class labels
class_labels =['0', '1', '10', '100', '101', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
st.title("Image Classifier")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.write('Please upload an image for classification')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        st.markdown('**Predictions**')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = dn_model.predict(img)  # make prediction using DenseNet model

        # Get the top 5 probabilities and their corresponding class labels
        top_5_indices = np.argsort(pred[0])[-5:][::-1]
        top_5_values = pred[0][top_5_indices]
        top_5_classes = [class_labels[i] for i in top_5_indices]

        # Create a DataFrame
        prediction = pd.DataFrame({
            'name': top_5_classes,
            'values': top_5_values
        })

        # Plot the results
        fig, ax = plt.subplots()
        ax = sns.barplot(y='name', x='values', data=prediction, order=prediction.sort_values('values', ascending=False).name)
        ax.set(xlabel='Confidence %', ylabel='Species')
        st.pyplot(fig)
