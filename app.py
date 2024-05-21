import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Load the model
model = load_model(r'C:\Users\DELL\Fruit_Veg_Classification\Image_Classification.keras')

data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
            'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
            'turnip', 'watermelon']

img_height = 180
img_width = 180

# File uploader widget to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    st.header('Image Classification Model')
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = np.expand_dims(img_arr, axis=0)  # Create batch dimension

    # Predict the class of the image
    predict = model.predict(img_bat)

    # Compute the scores
    score = tf.nn.softmax(predict[0])

    # Display the uploaded image and prediction results
    st.image(image_load, width=200)
    st.write('Veg/Fruit in image is {} with accuracy of {:0.2f}%'.format(data_cat[np.argmax(score)], np.max(score) * 100))
