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

# -------------------- LOAD DATA & PIPELINE -------------------- #
@st.cache_resource
def load_pipeline():
    pipeline = WaferCNNPipeline(
        model_path="cnn_model.keras",
        label_encoder_path="label_encoder.pkl"
    )
    return pipeline

pipeline = load_pipeline()

# Load wafer images (assuming already preprocessed in a list)
@st.cache_data
def load_wafer_images():
    # Replace with your actual data loading
    import glob
    import os
    files = glob.glob(os.path.join("wafer_images", "*"))
    images = []
    for f in files:
        ext = f.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "npy"]:
            if ext == "npy":
                arr = np.load(f)
                img = Image.fromarray((arr*255).astype(np.uint8))
            else:
                img = Image.open(f)
            images.append((f, img))
    return images

wafer_images = load_wafer_images()

# -------------------- IMAGE NAVIGATION -------------------- #
idx = st.slider("Select wafer image", 0, len(wafer_images)-1, 0)
img_name, wafer_img = wafer_images[idx]

st.subheader(f"Image: {img_name}")
st.image(wafer_img, caption=f"Wafer {idx}", use_column_width=True)

# -------------------- PREDICTION -------------------- #
pred_label, pred_probs = pipeline.predict(wafer_img)

st.subheader("Predicted Failure Type")
st.success(pred_label)

# -------------------- PROBABILITIES TABLE -------------------- #
prob_df = pd.DataFrame(
    list(pred_probs.items()), columns=["Failure Type", "Probability"]
).sort_values("Probability", ascending=False)

st.subheader("Prediction Probabilities")
st.dataframe(
    prob_df.style.format({"Probability": "{:.2f}"}).applymap(
        lambda x: "color: red" if x > 0.7 else ""
    ),
    height=300
)

# -------------------- BAR CHART -------------------- #
st.subheader("Probability Distribution")
st.bar_chart(prob_df.set_index("Failure Type")["Probability"])

# -------------------- INSIGHTS -------------------- #
st.subheader("Insights")
top2 = prob_df.head(2)
st.write(f"Top 2 likely failure types: {top2.iloc[0,0]} ({top2.iloc[0,1]:.2f}), "
         f"{top2.iloc[1,0]} ({top2.iloc[1,1]:.2f})")

if pred_probs.get("none", 0) > 0.7:
    st.info("Wafer looks mostly normal.")
elif pred_probs.get("[0 0]", 0) > 0.7:
    st.warning("Wafer may have no defects but check for anomalies.")
else:
    st.error("Potential defects detected. Review carefully!")

# -------------------- ABOUT -------------------- #
st.sidebar.header("About")
st.sidebar.info(
    """
    ChipSleuth is a semiconductor wafer defect detection dashboard.
    - Uses CNN trained with Focal Loss and Augmentation.
    - Displays predicted failure type and probabilities.
    - Allows navigation through wafer images.
    """
)