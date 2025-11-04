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
if "xgb_features" not in st.session_state:
    st.session_state.xgb_features = {}

# -------------------- PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = None  # Not used in new pipeline
DEMO_IMAGES = "demo_data/images"
XGB_MODEL_PATH = "xgboost_improved.pkl"
SCALER_PATH = "scaler.pkl"

# -------------------- LOAD MODELS -------------------- #
if os.path.exists(CNN_MODEL_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH)
else:
    st.error(f"CNN model not found: {CNN_MODEL_PATH}")
    cnn_pipe = None

if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
    xgb = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    xgb = None
    scaler = None
    st.warning("XGBoost model or scaler not found")

# -------------------- UTILITY FUNCTIONS -------------------- #

DEFECT_MAP = {
    0: 'No Defect', 1: 'Center', 2: 'Donut', 3: 'Edge-Ring',
    4: 'Scratch', 5: 'Near-full', 6: 'Random', 7: 'Local (Loc)', 8: 'Cluster'
}

def map_label(label):
    """Map numeric label to defect string."""
    try:
        label_key = int(label)
    except ValueError:
        label_key = str(label)
    return DEFECT_MAP.get(label_key, f"Unknown Defect ID: {label}")

def map_wafer_to_rgb(wafer_map):
    """Convert wafer map to RGB for display."""
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10, 10, 3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H, W = wafer_map.shape
    rgb_image = 50 * np.ones((H, W, 3), dtype=np.uint8)
    rgb_image[wafer_map == 1] = [0, 255, 255]  # Functional Die
    rgb_image[wafer_map == 2] = [255, 0, 0]    # Defect Die
    return rgb_image

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32, 32)):
    """Prepare wafer map for XGBoost model."""
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)
    if wafer_map.ndim != 2:
        if wafer_map.ndim == 3 and wafer_map.shape[2] == 3:
            wafer_map = np.array(Image.fromarray(wafer_map, 'RGB').convert('L'))
        else:
            if wafer_map.ndim > 2 and wafer_map.size == target_dim[0]*target_dim[1]:
                wafer_map = wafer_map.reshape(target_dim)
            else:
                return np.zeros(required_size)
    wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size:
        return np.pad(X_flat, (0, required_size-current_size), 'constant', constant_values=0)
    else:
        return X_flat[:required_size]

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    if model_choice == "CNN (Image-Based)" and cnn_pipe:
        st.subheader("Upload wafer images (.npy or .png) to run prediction")
        uploaded_files = st.file_uploader(
            "Upload wafer maps (.png, .jpg, .jpeg, .npy)", 
            type=["png","jpg","jpeg","npy"], accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    img = Image.open(uploaded_file).convert("L")
                    wafer = np.array(img)

                label, probs = cnn_pipe.predict(wafer)
                results.append({
                    "File": uploaded_file.name,
                    "Predicted_Label": label,
                    "Probabilities": probs,
                    "Wafer_Data": wafer
                })
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        if st.session_state.cnn_results:
            idx = st.session_state.cnn_index
            r = st.session_state.cnn_results[idx]
            wafer_rgb_display = map_wafer_to_rgb(r["Wafer_Data"])
            st.image(wafer_rgb_display, width=200, caption=f"Wafer Map: {r['File']}")
            st.markdown(f"**Predicted:** {r['Predicted_Label']}")

            st.subheader("Top Probabilities")
            top_probs = sorted(r['Probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
            for lbl, prob in top_probs:
                st.progress(np.clip(prob, 0.0, 1.0))
                st.markdown(f"**{lbl}**: {prob:.2f}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous"):
                    st.session_state.cnn_index = max(0, st.session_state.cnn_index - 1)
            with col2:
                if st.button("Next"):
                    st.session_state.cnn_index = min(len(st.session_state.cnn_results)-1, st.session_state.cnn_index + 1)

    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.warning("All uploaded images are resized to 32x32 and padded to 1029 features to match XGBoost training input.")
        uploaded_files = st.file_uploader(
            "Upload feature arrays (.npy) or images (.png, .jpg, .jpeg)", 
            type=["npy", "png", "jpg", "jpeg"], accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            for file in uploaded_files:
                X = None
                wafer = None
                try:
                    if file.name.endswith(".npy"):
                        wafer = np.load(file)
                    else:
                        img = Image.open(file).convert("L")
                        wafer = np.array(img)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
                    continue

                if wafer is not None:
                    X = prepare_pixel_features_for_xgb(wafer).reshape(1, -1)

                try:
                    X_scaled = scaler.transform(X)
                    pred = int(xgb.predict(X_scaled)[0])
                    results.append({"File": file.name, "Predicted_Label": map_label(pred)})
                except Exception as e:
                    st.error(f"Prediction error for {file.name}: {e}")

            st.session_state.xgb_results = results
            st.subheader("XGBoost Predictions:")
            for r in results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights for Current Prediction")
    if st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        current_file = r['File']
        predicted_label = r['Predicted_Label']
        probs = r['Probabilities']

        st.markdown(f"### Analysis for Wafer: **{current_file}**")
        st.markdown(f"**Primary Prediction:** {predicted_label}")

        prob_df = pd.DataFrame(
            {'Defect Type': list(probs.keys()), 'Confidence': list(probs.values())}
        ).sort_values(by='Confidence', ascending=False)
        st.bar_chart(prob_df.set_index('Defect Type'), height=350)
    else:
        st.warning("Please upload a wafer image and run a prediction first.")

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
    st.markdown("A typical wafer map highlights defect regions against the functional wafer area:")
    st.image("https://placehold.co/600x600/1e293b/f8fafc?text=Example+Wafer+Map+with+Defects", width=300)
