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

st.title("🚀 ChipSleuth – Wafer Defect Prediction")

# -------------------- LOAD MODEL -------------------- #
cnn_pipeline = WaferCNNPipeline(
    model_path="cnn_model.keras",
    label_encoder_path="label_encoder.pkl",
    class_weights=None
)

# -------------------- LOAD IMAGES FROM FOLDER -------------------- #
image_folder = "image_data"  # folder with .jpg, .jpeg, .png, .npy
wafer_images = []

for f in os.listdir(image_folder):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(image_folder, f))
        wafer_images.append((f, img))
    elif f.lower().endswith(".npy"):
        arr = np.load(os.path.join(image_folder, f))
        img = Image.fromarray(arr.astype(np.uint8))
        wafer_images.append((f, img))

# -------------------- UPLOAD NEW IMAGES -------------------- #
uploaded_files = st.sidebar.file_uploader(
    "📥 Upload your wafer images",
    type=["jpg", "jpeg", "png", "npy"],
    accept_multiple_files=True
)

for f in uploaded_files:
    if f.name.lower().endswith(".npy"):
        arr = np.load(f)
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.open(f)
    wafer_images.append((f.name, img))

if len(wafer_images) == 0:
    st.warning(" No images found or uploaded!")
    st.stop()

# -------------------- SIDEBAR -------------------- #
st.sidebar.title("🎛️ Controls")
view_mode = st.sidebar.radio("Select View Mode", ["Slider View", "Batch View"])

# -------------------- SLIDER VIEW -------------------- #
if view_mode == "Slider View":
    st.header("🖼️ Single Wafer View")
    idx = st.slider("Select Wafer Index", 0, len(wafer_images)-1, 0)
    img_name, wafer_img = wafer_images[idx]

    st.image(wafer_img, use_container_width=True, caption=img_name)

    # Prediction
    label, probs = cnn_pipeline.predict(wafer_img)
    st.subheader("🔍 Predicted Failure Type")
    st.write(f"**{label}**")

    # Top 5 probabilities
    st.subheader("📊 Top 5 Prediction Probabilities")
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for cls, p in top5:
        st.write(f"{cls}: {p:.2f}")

    # Insights
    st.subheader(" Insights")
    st.write(f"- Highest probability: **{top5[0][1]:.2f} ({top5[0][0]})**")
    low_prob_classes = [k for k, v in probs.items() if v < 0.05]
    st.write(f"- Low probability (<0.05) classes: {low_prob_classes}")

# -------------------- BATCH VIEW -------------------- #
else:
    st.header("⚙️ Batch Wafer View")
    batch_size = st.sidebar.slider("Images per row", 3, 8, 4)
    max_images = st.sidebar.slider("Max images to display", 4, len(wafer_images), 20)

    for start_idx in range(0, min(len(wafer_images), max_images), batch_size):
        cols = st.columns(batch_size)
        for col_idx, (img_name, wafer_img) in enumerate(
            wafer_images[start_idx:start_idx + batch_size]
        ):
            col = cols[col_idx]
            col.image(wafer_img, use_container_width=True, caption=img_name)
            label, probs = cnn_pipeline.predict(wafer_img)
            top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            col.markdown(f"**Predicted:** {top5[0][0]} ({top5[0][1]:.2f})")
            for cls, p in top5[1:]:
                col.markdown(f"{cls}: {p:.2f}")

# -------------------- COLLECT ALL PREDICTIONS -------------------- #
all_results = []
for img_name, wafer_img in wafer_images:
    label, probs = cnn_pipeline.predict(wafer_img)
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    result = {"Image": img_name}
    for i, (cls, p) in enumerate(top5, 1):
        result[f"Top{i}_Class"] = cls
        result[f"Top{i}_Prob"] = round(p, 4)
    all_results.append(result)

df_results = pd.DataFrame(all_results)

# -------------------- DOWNLOAD BUTTON -------------------- #
st.markdown("---")
st.subheader("📥 Download Predictions")
csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="🎯 Download CSV",
    data=csv,
    file_name="wafer_predictions.csv",
    mime="text/csv"
)

# -------------------- ABOUT -------------------- #
st.sidebar.markdown("---")
st.sidebar.header(" About")
st.sidebar.info(
    """
    **ChipSleuth – Semiconductor Wafer Defect Detection**

    - Upload wafer images in `.jpg`, `.jpeg`, `.png`, or `.npy`.
    - **Slider View:** Inspect single wafer predictions with insights.
    - **Batch View:** View multiple wafers side-by-side.
    - **Download CSV:** Export all predictions with top-5 probabilities.
    - Built on CNN trained with focal loss, class weights, and heavy augmentation.
    """
)