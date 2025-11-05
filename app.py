import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pandas as pd
from cnn_pipeline import WaferCNNPipeline 

# -------------------- STREAMLIT CONFIG -------------------- #
st.set_page_config(
    page_title="ChipSleuth Wafer Defect Dashboard",
    page_icon="🕵️‍♀️",
    layout="wide"
)

st.title("ChipSleuth – Wafer Defect Prediction")

# -------------------- LOAD MODEL -------------------- #
model_path = "cnn_model.keras"
le_path = "label_encoder.pkl"
pipeline = WaferCNNPipeline(model_path, le_path)

# -------------------- LOAD IMAGES -------------------- #
image_folder = "image_data"  # Folder with .jpg, .jpeg, .png, .npy
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.npy'))]
wafer_images = []

for fname in image_files:
    path = os.path.join(image_folder, fname)
    if fname.lower().endswith('.npy'):
        img = np.load(path)
    else:
        img = Image.open(path)
    wafer_images.append((fname, img))

if not wafer_images:
    st.warning("No wafer images found in the folder.")
    st.stop()

# -------------------- SLIDER -------------------- #
idx = st.slider("Select Wafer Image", 0, len(wafer_images)-1, 0)
img_name, wafer_img = wafer_images[idx]

# -------------------- DISPLAY IMAGE -------------------- #
if isinstance(wafer_img, np.ndarray):
    wafer_display = Image.fromarray((wafer_img * 255).astype(np.uint8)) if wafer_img.max() <= 1.0 else Image.fromarray(wafer_img.astype(np.uint8))
else:
    wafer_display = wafer_img

st.image(wafer_display, caption=img_name, width=300)

# -------------------- PREDICTION -------------------- #
pred_label, pred_probs = pipeline.predict(wafer_img)
st.subheader("Predicted Failure Type")
st.write(f"**{pred_label}**")

# Show probabilities
st.subheader("Prediction Probabilities")
for k,v in pred_probs.items():
    st.write(f"{k}: {v:.3f}")

# -------------------- INSIGHTS -------------------- #
st.subheader("Insights")
st.write("""
- The model uses a CNN trained on normalized wafer maps.
- Probabilities reflect confidence per failure type.
- Dark areas in the wafer map often correspond to defects detected by the model.
""")

# -------------------- ABOUT -------------------- #
st.subheader("About")
st.write("""
ChipSleuth is a demo for semiconductor wafer defect detection.
The CNN model classifies wafers into failure types based on image patterns.
""")