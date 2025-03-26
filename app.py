import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
MODEL_PATH = "malaria_prediction_model1.h5"
model = load_model(MODEL_PATH)

# Define image dimensions based on model input size
img_height, img_width = 150, 150

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b7fcb;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        color: #00d4ff;
        font-weight: bold;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #222;
    }
    .stButton>button {
        background-color: #00ff88;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction function
def predict_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_class = "Parasitized" if prediction[0][0] < 0.5 else "Uninfected"
    return predicted_class, prediction[0][0]

# Streamlit UI
st.markdown('<h1 class="main-title">Malaria Detection using Deep Learning</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload a cell image to check if it is Parasitized or Uninfected.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=500)

    # Save the uploaded image to a temporary location
    temp_image_path = "temp_uploaded_image.png"
    img.save(temp_image_path)

    # Add a "Predict" button
    if st.button("Predict"):
        # Make prediction
        predicted_label, raw_output = predict_image(temp_image_path)

        # Display results
        st.markdown(f"<h2 style='color:#ff4b4b;'>Predicted Class: {predicted_label}</h2>", unsafe_allow_html=True)
        #st.markdown(f"<h3 style='color:#444;'>Model Confidence: {raw_output:.4f}</h3>", unsafe_allow_html=True)
