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
cnn_pipeline = WaferCNNPipeline(
    model_path="cnn_model.keras",
    label_encoder_path="label_encoder.pkl",
    class_weights=None
)

# -------------------- LOAD IMAGES -------------------- #
image_folder = "image_data"  # folder with .jpg, .jpeg, .png, .npy
wafer_images = []

for f in os.listdir(image_folder):
    file_path = os.path.join(image_folder, f)
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(file_path)
        wafer_images.append((f, img))
    elif f.lower().endswith(".npy"):
        arr = np.load(file_path)
        # Normalize for display
        arr_disp = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255
        img = Image.fromarray(arr_disp.astype(np.uint8))
        wafer_images.append((f, img))

# -------------------- UPLOAD NEW FILES -------------------- #
uploaded_files = st.sidebar.file_uploader(
    "Upload Wafer Images (.jpg, .jpeg, .png, .npy)",
    type=["jpg","jpeg","png","npy"],
    accept_multiple_files=True
)
for file in uploaded_files:
    if file.name.lower().endswith(".npy"):
        arr = np.load(file)
        arr_disp = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255
        img = Image.fromarray(arr_disp.astype(np.uint8))
    else:
        img = Image.open(file)
    wafer_images.append((file.name, img))

if len(wafer_images) == 0:
    st.warning("No images found or uploaded!")
    st.stop()

# -------------------- SIDEBAR -------------------- #
st.sidebar.title("Controls")
view_mode = st.sidebar.radio("Select View Mode", ["Slider View", "Batch View"])

# -------------------- SLIDER VIEW -------------------- #
if view_mode == "Slider View":
    st.header("🖼️ Single Wafer View")
    idx = st.slider("Select Wafer Index", 0, len(wafer_images)-1, 0)
    img_name, wafer_img = wafer_images[idx]

    # Display properly normalized
    display_img = wafer_img.copy()
    st.image(display_img, use_column_width=True, caption=img_name)

    # Prediction
    label, probs = cnn_pipeline.predict(wafer_img)
    st.subheader("Predicted Failure Type")
    st.write(f"**{label}**")

    st.subheader("Prediction Probabilities")
    for k, v in probs.items():
        st.write(f"{k}: {v:.2f}")

    # Insights
    st.subheader("Insights")
    st.write(f"- Highest probability: {max(probs.values()):.2f}")
    st.write(f"- Classes with low probability (<0.05): {[k for k,v in probs.items() if v<0.05]}")

# -------------------- BATCH VIEW -------------------- #
else:
    st.header("📊 Batch Wafer View")
    batch_size = st.sidebar.slider("Images per row", 3, 8, 4)
    max_images = st.sidebar.slider("Max images to display", 4, len(wafer_images), 20)

    for start_idx in range(0, min(len(wafer_images), max_images), batch_size):
        cols = st.columns(batch_size)
        for col_idx, (img_name, wafer_img) in enumerate(
                wafer_images[start_idx:start_idx + batch_size]):
            col = cols[col_idx]

            # Display properly normalized
            arr = np.array(wafer_img)
            arr_disp = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255
            display_img = Image.fromarray(arr_disp.astype(np.uint8))
            col.image(display_img, use_column_width=True, caption=img_name)

            # Predict
            label, probs = cnn_pipeline.predict(wafer_img)
            top_class = max(probs, key=probs.get)
            top_prob = probs[top_class]
            col.markdown(f"**Predicted:** {top_class} ({top_prob:.2f})")

# -------------------- ABOUT -------------------- #
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    """
    ChipSleuth – Semiconductor Wafer Defect Detection

    - Upload your wafer images in `.jpg`, `.jpeg`, `.png`, or `.npy` format.
    - Slider view: Inspect one wafer at a time with full probabilities and insights.
    - Batch view: Preview multiple wafers with top predictions.
    - Powered by CNN trained with focal loss, class weights, and augmentation.
    """
)