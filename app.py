import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# -------------------- IMPORTS -------------------- #
import os
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import logging
from cnn_pipeline import WaferCNNPipeline

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger = lambda: logging.getLogger('tensorflow')
tf.get_logger().setLevel(logging.ERROR)

# -------------------- STREAMLIT CONFIG -------------------- #
st.set_page_config(
    page_title="ChipSleuth",
    page_icon="🕵️‍♀️",
    layout="wide"
)
st.title("ChipSleuth – Semiconductor Wafer Defect Detection")

# -------------------- SESSION STATE -------------------- #
if "cnn_results" not in st.session_state:
    st.session_state.cnn_results = []
if "cnn_index" not in st.session_state:
    st.session_state.cnn_index = 0
if "xgb_results" not in st.session_state:
    st.session_state.xgb_results = []

# -------------------- FILE PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
XGB_MODEL_PATH = "xgboost_improved.pkl"

# -------------------- LOAD MODELS -------------------- #
if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.error("CNN model or label encoder not found.")
    cnn_pipe = None

if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
    xgb = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("XGBoost model or scaler not found.")
    xgb = None
    scaler = None

# -------------------- UTILITY -------------------- #
def map_label(label):
    return "No Defect" if label in [0, "0 0", "0"] else str(label)

def display_cnn_image(idx):
    if st.session_state.cnn_results:
        r = st.session_state.cnn_results[idx]
        wafer_file = r["File"]
        wafer = np.load(wafer_file) if wafer_file.endswith(".npy") else np.array(Image.open(wafer_file).convert("L"))
        img_to_show = (wafer * 255).astype(np.uint8)
        st.image(img_to_show, width=300, clamp=True, channels="L")
        st.markdown(f"**Predicted Defect:** {r['Predicted_Label']}")
        # Show class probabilities
        for cls, prob in r["Probabilities"].items():
            st.progress(int(prob*100), text=cls)

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    # -------------------- CNN -------------------- #
    if model_choice == "CNN (Image-Based)" and cnn_pipe:
        uploaded_files = st.file_uploader(
            "Upload wafer images (.png, .jpg, .jpeg, .npy)", type=["png","jpg","jpeg","npy"], accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.cnn_results = []
            for f in uploaded_files:
                wafer = np.load(f) if f.name.endswith(".npy") else np.array(Image.open(f).convert("L"))
                label, probs = cnn_pipe.predict(wafer)
                st.session_state.cnn_results.append({
                    "File": f.name,
                    "Predicted_Label": map_label(label),
                    "Probabilities": probs
                })
            st.session_state.cnn_index = 0

        if st.session_state.cnn_results:
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                if st.button("Previous"):
                    st.session_state.cnn_index = max(st.session_state.cnn_index-1, 0)
            with col2:
                display_cnn_image(st.session_state.cnn_index)
            with col3:
                if st.button("Next"):
                    st.session_state.cnn_index = min(st.session_state.cnn_index+1, len(st.session_state.cnn_results)-1)

    # -------------------- XGBoost -------------------- #
    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.subheader("Drag `.npy` feature arrays (10 features each) here")
        uploaded_files = st.file_uploader(
            "Upload feature arrays (.npy)", type=["npy"], accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.xgb_results = []
            for f in uploaded_files:
                X = np.load(f).reshape(1,-1)
                try:
                    X_scaled = scaler.transform(X)
                    pred = xgb.predict(X_scaled)[0]
                    st.session_state.xgb_results.append({
                        "File": f.name,
                        "Predicted_Label": map_label(pred)
                    })
                except ValueError as e:
                    st.error(f"Feature mismatch for {f.name}: {e}")
            for r in st.session_state.xgb_results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: ABOUT -------------------- #
with tabs[1]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps  
    - **XGBoost** for feature-based wafer data  
    - **Streamlit** for an interactive dashboard deployment  

    **Goal:** Automate defect detection and enhance wafer yield prediction.
    """)