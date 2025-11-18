import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.title("🌿 Plant Disease Classifier")

# Load model and classes
model = tf.keras.models.load_model('models/plant_disease_model.keras')
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

st.write("Upload a plant leaf image")

# Upload image
uploaded_file = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    
    # Classify
    if st.button('Classify', type="primary"):
        # Prepare image
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        pred = model.predict(img_array)
        disease_idx = np.argmax(pred[0])
        disease = class_names[disease_idx] 
        confidence = pred[0][disease_idx] * 100
        
        # Show result
        st.success(f"Disease: {disease}")
        st.info(f"Confidence: {confidence:.1f}%")