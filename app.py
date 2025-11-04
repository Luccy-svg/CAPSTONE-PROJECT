import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# -------------------- IMPORTS -------------------- #
import streamlit as st
import numpy as np
from PIL import Image
import joblib
from cnn_pipeline import WaferCNNPipeline

# -------------------- STREAMLIT CONFIG -------------------- #
st.set_page_config(
    page_title="ChipSleuth: Wafer Defect Dashboard",
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

# -------------------- PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = "demo_data/label_encoder.pkl"
DEMO_IMAGES = "demo_data/images"

# -------------------- LOAD MODELS -------------------- #
if os.path.exists(CNN_MODEL_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.error(f"CNN model not found: {CNN_MODEL_PATH}")
    cnn_pipe = None

xgb_model_path = "xgboost_improved.pkl"
scaler_path = "scaler.pkl"

if os.path.exists(xgb_model_path):
    xgb = joblib.load(xgb_model_path)
    scaler = joblib.load(scaler_path)
else:
    st.warning("XGBoost model or scaler not found")
    xgb = None
    scaler = None

# -------------------- UTILITY -------------------- #
def map_label(label):
    return "No Defect" if label == "0 0" else label

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    # -------------------- CNN INTERACTIVE -------------------- #
    if model_choice == "CNN (Image-Based)":
        st.subheader("Upload wafer images or use demo images")
        uploaded_files = st.file_uploader(
            "Upload wafer maps (.png, .jpg, .jpeg, .npy)", 
            type=["png","jpg","jpeg","npy"], accept_multiple_files=True
        )

        # Load demo if no upload
        if st.button("Load Demo Images") and cnn_pipe:
            demo_files = sorted([os.path.join(DEMO_IMAGES, f) for f in os.listdir(DEMO_IMAGES) if f.endswith(".npy")])
            results = []
            for file in demo_files:
                wafer = np.load(file)
                label, probs = cnn_pipe.predict(wafer)
                results.append({"File": os.path.basename(file), "Predicted_Label": label, "Probabilities": probs})
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Predict uploaded
        if uploaded_files and cnn_pipe:
            results = []
            for uploaded_file in uploaded_files:
                wafer = np.load(uploaded_file) if uploaded_file.name.endswith(".npy") else Image.open(uploaded_file).convert("L")
                label, probs = cnn_pipe.predict(wafer)
                results.append({"File": uploaded_file.name, "Predicted_Label": label, "Probabilities": probs})
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Display interactive image
        if st.session_state.cnn_results:
            idx = st.session_state.cnn_index
            r = st.session_state.cnn_results[idx]

            # Load wafer image (demo or uploaded)
            wafer_file = os.path.join(DEMO_IMAGES, r["File"])
            if os.path.exists(wafer_file):
                wafer = np.load(wafer_file)

                # Scale to 0-255 and convert to RGB for display
                wafer_display = (wafer / wafer.max() * 255).astype(np.uint8)
                wafer_rgb = np.stack([wafer_display]*3, axis=-1)
                st.image(wafer_rgb, width=200)

            st.markdown(f"**Predicted:** {r['Predicted_Label']}")

            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous"):
                    st.session_state.cnn_index = max(0, st.session_state.cnn_index - 1)
            with col2:
                if st.button("Next"):
                    st.session_state.cnn_index = min(len(st.session_state.cnn_results)-1, st.session_state.cnn_index + 1)

    # -------------------- XGBOOST DRAG ONLY -------------------- #
    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.subheader("Drag `.npy` feature arrays to predict (10 features per wafer)")
        uploaded_files = st.file_uploader(
            "Upload feature arrays (.npy)", type=["npy"], accept_multiple_files=True
        )
        if uploaded_files:
            results = []
            for file in uploaded_files:
                X = np.load(file).reshape(1,-1)
                try:
                    X_scaled = scaler.transform(X)
                    pred = xgb.predict(X_scaled)[0]
                    results.append({"File": file.name, "Predicted_Label": map_label(str(pred))})
                except ValueError as e:
                    st.error(f"Feature mismatch for {file.name}: {e}")
            st.session_state.xgb_results = results
            for r in results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights")
    st.write("CNN and XGBoost insights will be added here. (Confusion matrices, probabilities, etc.)")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps  
    - **XGBoost** for feature-based wafer data  
    - **Streamlit** for interactive dashboard deployment  

    **Goal:** Automate defect detection and enhance wafer yield prediction.
    """)