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



# STREAMLIT CONFIG

st.set_page_config(
    page_title="ChipSleuth Wafer Defect Dashboard",
    page_icon="🕵️‍♀️",
    layout="wide"
)
st.title("🚀 ChipSleuth – Wafer Defect Prediction")


# LOAD MODEL

cnn_pipeline = WaferCNNPipeline(
    model_path="cnn_model.keras",
    label_encoder_path="label_encoder.pkl",
    class_weights=None
)

if cnn_pipeline is None:
    st.error("CNN model failed to load! Please check cnn_model.keras and label_encoder.pkl paths.")
    st.stop()
else:
    st.sidebar.success("CNN model loaded successfully.")


# LOAD IMAGES FROM FOLDERS

if os.path.exists("image_data") and len(os.listdir("image_data")) > 0:
    image_folder = "image_data"
elif os.path.exists("demo_images") and len(os.listdir("demo_images")) > 0:
    image_folder = "demo_images"
else:
    image_folder = None

wafer_images = []
if image_folder and os.path.exists(image_folder):
    for f in os.listdir(image_folder):
        file_path = os.path.join(image_folder, f)
        try:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                color_img = Image.open(file_path).copy()
                gray_img = color_img.convert("L")
                wafer_images.append((f, color_img, gray_img))
            elif f.lower().endswith(".npy"):
                arr = np.load(file_path, allow_pickle=True)
                if arr.ndim > 2:
                    arr = np.mean(arr, axis=-1)
                img = Image.fromarray(arr.astype(np.uint8))
                wafer_images.append((f, img, img))
        except Exception as e:
            st.warning(f" Could not load {f}: {e}")
else:
    st.warning(f"No image folder found: {image_folder}")


# UPLOAD NEW IMAGES

st.sidebar.markdown("### Upload New Wafer Images")
uploaded_files = st.sidebar.file_uploader(
    "Upload your wafer images",
    type=["jpg", "jpeg", "png", "npy"],
    accept_multiple_files=True
)

for f in uploaded_files:
    try:
        if f.name.lower().endswith(".npy"):
            arr = np.load(f, allow_pickle=True)
            if arr.ndim > 2:
                arr = np.mean(arr, axis=-1)
            img = Image.fromarray(arr.astype(np.uint8))
            wafer_images.append((f.name, img, img))
        else:
            color_img = Image.open(f).copy()
            gray_img = color_img.convert("L")
            wafer_images.append((f.name, color_img, gray_img))
    except Exception as e:
        st.sidebar.error(f"Failed to read {f.name}: {e}")

# Sidebar preview of uploaded images
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    st.sidebar.markdown("### Uploaded Previews")
    cols = st.sidebar.columns(min(3, len(uploaded_files)))
    for i, (name, color_img, _) in enumerate(wafer_images[-len(uploaded_files):]):
        cols[i % len(cols)].image(color_img, caption=name, use_container_width=True)

# Stop if no images
if len(wafer_images) == 0:
    st.warning("🔩 No images found or uploaded! Please upload .png, .jpg, or .npy wafer maps.")
    st.stop()

st.sidebar.info(f"🖼️ Total images loaded: **{len(wafer_images)}**")


# SIDEBAR CONTROLS

st.sidebar.title("🔧 Controls")
view_mode = st.sidebar.radio("Select View Mode", ["Slider View", "Batch View"])


# SLIDER VIEW

if view_mode == "Slider View":
    st.header("🖼️ Single Wafer View")

    default_idx = len(wafer_images) - 1 if uploaded_files else 0
    idx = st.slider("Select Wafer Index", 0, len(wafer_images)-1, default_idx)

    img_name, color_img, gray_img = wafer_images[idx]
    st.image(color_img, width=500, caption=img_name)

    # Prediction
    label, probs = cnn_pipeline.predict(gray_img)
    st.subheader("🔍 Predicted Failure Type")
    st.write(f"**{label}**")

    # Top 5 probabilities
    st.subheader("📊 Top 5 Prediction Probabilities")
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for cls, p in top5:
        st.write(f"{cls}: {p:.2f}")

    # Insights
    st.subheader("Insights")
    st.write(f"- Highest probability: **{top5[0][1]:.2f} ({top5[0][0]})**")
    low_prob_classes = [k for k, v in probs.items() if v < 0.05]
    st.write(f"- Low probability (<0.05) classes: {low_prob_classes}")


# BATCH VIEW

else:
    st.header("⚙️ Batch Wafer View")
    batch_size = st.sidebar.slider("Images per row", 3, 8, 4)
    max_images = st.sidebar.slider("Max images to display", 4, len(wafer_images), min(20, len(wafer_images)))

    for start_idx in range(0, min(len(wafer_images), max_images), batch_size):
        cols = st.columns(batch_size)
        for col_idx, (img_name, color_img, gray_img) in enumerate(
            wafer_images[start_idx:start_idx + batch_size]
        ):
            col = cols[col_idx]
            col.image(color_img, use_container_width=True, caption=img_name)
            label, probs = cnn_pipeline.predict(gray_img)
            top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            col.markdown(f"**Predicted:** {top5[0][0]} ({top5[0][1]:.2f})")
            for cls, p in top5[1:]:
                col.markdown(f"{cls}: {p:.2f}")


# COLLECT ALL PREDICTIONS

all_results = []
for img_name, _, gray_img in wafer_images:
    label, probs = cnn_pipeline.predict(gray_img)
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    result = {"Image": img_name}
    for i, (cls, p) in enumerate(top5, 1):
        result[f"Top{i}_Class"] = cls
        result[f"Top{i}_Prob"] = round(p, 4)
    all_results.append(result)

df_results = pd.DataFrame(all_results)


# DOWNLOAD RESULTS

st.markdown("---")
st.subheader("Download Predictions")
csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="🎯 Download CSV",
    data=csv,
    file_name="wafer_predictions.csv",
    mime="text/csv"
)


# ABOUT SECTION

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