#Import
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
from PIL import Image

model_path_to_metadata = "deepfake_detection_system_model.joblib"
# Function to load the metadat and from it the model, and class_names, made a function to boost code reusability
@st.cache_resource
def load_model_and_metadata():
    # Load metadata
    metadata = joblib.load(model_path_to_metadata)
    class_names = metadata["class_names"]

    # Load trained CNN model
    model = load_model(metadata["model_path"])

    return model, class_names

model, class_names = load_model_and_metadata()

#Setting up the streamlit UI
st.set_page_config(page_title="Deepfake Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ Deepfake Detection System")
st.write("Upload an image and the model will classify it as **Real** or **Deepfake**.")

#Setting up a file uploader so that user can upload the file, and then it can be sent to model
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    #Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    #Preprocessint image to make it usable by the model
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)

    #Prediction
    prediction = model.predict(img_array)[0][0]  # sigmoid output
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    #Map prediction
    predicted_class = class_names[1] if prediction > 0.5 else class_names[0]

    # Display result
    st.subheader("🔍 Prediction Result:")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")