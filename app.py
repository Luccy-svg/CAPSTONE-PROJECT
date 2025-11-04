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

# -------------------- PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
XGB_MODEL_PATH = "xgboost_improved.pkl"
SCALER_PATH = "scaler.pkl"

# -------------------- DEFECT LABEL MAPPING -------------------- #
mapping_type = {
    0: 'Center', 1: 'Donut', 2: 'Edge-Loc', 3: 'Edge-Ring',
    4: 'Loc', 5: 'Random', 6: 'Scratch', 7: 'none', 8: 'Unknown'
}

# -------------------- LOAD MODELS -------------------- #
cnn_pipe = None
if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.warning("CNN model or label encoder not found!")

xgb = None
scaler = None
if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
    xgb = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.warning("XGBoost model or scaler not found!")

# -------------------- UTILITY FUNCTIONS -------------------- #
def map_label(label_idx):
    return mapping_type.get(label_idx, "Unknown")

def map_wafer_to_rgb(wafer_map):
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10, 10, 3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H, W = wafer_map.shape
    rgb_image = 50 * np.ones((H, W, 3), dtype=np.uint8)
    rgb_image[wafer_map == 1] = [0, 255, 255]
    rgb_image[wafer_map == 2] = [255, 0, 0]
    return rgb_image

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32, 32)):
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)
    if wafer_map.ndim != 2:
        wafer_map = np.array(Image.fromarray(wafer_map).convert('L'))
    wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size:
        padding_needed = required_size - current_size
        return np.pad(X_flat, (0, padding_needed), 'constant', constant_values=0)
    else:
        return X_flat[:required_size]

# -------------------- STREAMLIT TABS -------------------- #
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
                    wafer = np.array(Image.open(uploaded_file).convert("L"))
                label, probs = cnn_pipe.predict(wafer)
                results.append({"File": uploaded_file.name, "Predicted_Label": label, "Probabilities": probs, "Wafer_Data": wafer})
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        if st.session_state.cnn_results:
            idx = st.session_state.cnn_index
            r = st.session_state.cnn_results[idx]
            wafer_rgb_display = map_wafer_to_rgb(r['Wafer_Data'])
            st.image(wafer_rgb_display, width=200, caption=f"Wafer Map: {r['File']}")
            st.markdown(f"**Predicted:** {r['Predicted_Label']}")
            st.subheader("Prediction Probabilities")
            probs = r['Probabilities']
            if isinstance(probs, dict):
                for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.progress(np.clip(prob, 0.0, 1.0))
                    st.markdown(f"**{label}**: {prob:.2f}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous"):
                    st.session_state.cnn_index = max(0, st.session_state.cnn_index - 1)
            with col2:
                if st.button("Next"):
                    st.session_state.cnn_index = min(len(st.session_state.cnn_results)-1, st.session_state.cnn_index + 1)

    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.warning("All images resized to 32x32 and padded to 1029 features for prediction.")
        uploaded_files = st.file_uploader(
            "Upload feature arrays (.npy) or images (.png, .jpg, .jpeg)", 
            type=["npy","png","jpg","jpeg"], accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer = None
                if file.name.endswith(".npy"):
                    wafer = np.load(file)
                else:
                    wafer = np.array(Image.open(file).convert("L"))
                X_features = prepare_pixel_features_for_xgb(wafer).reshape(1, -1)
                X_scaled = scaler.transform(X_features)
                pred_idx = int(xgb.predict(X_scaled)[0])
                pred_label = map_label(pred_idx)
                results.append({"File": file.name, "Predicted_Label": pred_label})
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
        st.markdown(f"### Wafer: {r['File']}")
        st.markdown(f"**Primary Prediction:** {r['Predicted_Label']}")
        probs = r['Probabilities']
        if isinstance(probs, dict):
            prob_df = pd.DataFrame({'Defect Type':[k for k in probs.keys()], 'Confidence':[v for v in probs.values()]})
            prob_df = prob_df.sort_values('Confidence', ascending=False)
            st.bar_chart(prob_df.set_index('Defect Type'))
    else:
        st.warning("Please upload a wafer image or run a prediction on the 'Predict Defects' tab first.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps
    - **XGBoost** for feature-based wafer data
    - **Streamlit** for interactive dashboard deployment
    """)
    st.image("https://placehold.co/600x600/1e293b/f8fafc?text=Example+Wafer+Map+with+Defects", width=300)
