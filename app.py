import os
os.environ["KERAS_BACKEND"] = "jax"  # Use JAX backend instead of TensorFlow

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import keras
import gdown

# Page config
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

.stApp {
    background-color: #0a0a0a;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}

.title-block {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.title-block h1 {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: #666;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.upload-area {
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 2rem;
    background: #111;
    margin: 1.5rem 0;
}

.result-card {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-positive {
    background: #1a0a0a;
    border: 1px solid #ff3b3b;
}
.result-negative {
    background: #0a1a0a;
    border: 1px solid #22c55e;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-positive .result-label { color: #ff3b3b; }
.result-negative .result-label { color: #22c55e; }

.result-sub {
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
}

hr {
    border: none;
    border-top: 1px solid #1f1f1f;
    margin: 2rem 0;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stFileUploader"] {
    background: #111;
}
[data-testid="stFileUploader"] section {
    background: #111;
    border: 1px dashed #333;
    border-radius: 8px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #555;
}

.stButton > button {
    background: #ffffff;
    color: #000000;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #d4d4d4;
    color: #000;
}

.loading-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    gap: 1rem;
}
.spinner-ring {
    width: 48px;
    height: 48px;
    border: 3px solid #222;
    border-top: 3px solid #ffffff;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}
@keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-text {
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1; }
}

.warning-box {
    background: #111800;
    border: 1px solid #3a3a00;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-block">
    <h1>🧠 Brain Tumor Detector</h1>
    <p>MRI Classification · CNN Model</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "brain_tumor_model.h5"
    try:
        if not os.path.exists(model_path):
            file_id = "1Pc3vgDNR_KZDa6TwiKvLqcT7D0XjH-rf"
            url = f"https://drive.google.com/uc?id={file_id}"
            with st.spinner("Downloading AI model... Please wait."):
                gdown.download(url, model_path, quiet=False)

        model = keras.saving.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Upload
st.markdown("#### Upload MRI Scan")

uploaded_file = st.file_uploader(
    label="Upload MRI Image",
    label_visibility="collapsed",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image"
)

if uploaded_file is not None:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="loading-box">
        <div class="spinner-ring"></div>
        <div class="loading-text">Loading scan...</div>
    </div>
    """, unsafe_allow_html=True)

    image = Image.open(uploaded_file)
    _ = np.array(image)

    loading_placeholder.empty()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("ANALYSE SCAN"):
        with st.spinner("Analysing..."):
            img_array = np.array(image.convert("RGB"))
            img_resized = cv2.resize(img_array, (200, 200))
            img_normalized = img_resized.astype("float32") / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            prediction = model.predict(img_input, verbose=0)[0][0]
            has_tumor = float(prediction) > 0.5

        if has_tumor:
            st.markdown("""
            <div class="result-card result-positive">
                <div class="result-label">TUMOR DETECTED</div>
                <div class="result-sub">Positive · Please consult a specialist</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-negative">
                <div class="result-label">NO TUMOR FOUND</div>
                <div class="result-sub">Negative · No anomaly detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            ⚠️ This tool is for educational purposes only and is not a substitute
            for professional medical diagnosis. Always consult a qualified medical
            professional for clinical decisions.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; color:#444; font-size:0.82rem; 
                letter-spacing:1px; padding: 2rem 0;">
        UPLOAD AN MRI IMAGE TO BEGIN
    </div>
    """, unsafe_allow_html=True)