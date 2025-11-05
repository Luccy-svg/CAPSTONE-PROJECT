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
if "cnn_results" not in st.session_state: st.session_state.cnn_results = []
if "cnn_index" not in st.session_state: st.session_state.cnn_index = 0
if "xgb_results" not in st.session_state: st.session_state.xgb_results = []

# -------------------- PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
XGB_MODEL_PATH = "xgboost_improved.pkl"
SCALER_PATH = "scaler.pkl"

# -------------------- LOAD MODELS -------------------- #
@st.cache_resource
def load_cnn_pipeline():
    if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        try:
            return WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
        except Exception as e:
            st.error(f"Error loading CNN Pipeline: {e}. Ensure Focal Loss is handled in cnn_pipeline.py")
    st.warning("CNN model or label encoder not found.")
    return None

@st.cache_resource
def load_xgb_models():
    if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            xgb_model = joblib.load(XGB_MODEL_PATH)
            scaler_model = joblib.load(SCALER_PATH)
            return xgb_model, scaler_model
        except Exception as e:
            st.error(f"Error loading XGBoost models: {e}")
    st.warning("XGBoost model or scaler not found.")
    return None, None

cnn_pipe = load_cnn_pipeline()
xgb, scaler = load_xgb_models()

# -------------------- DEFECT MAPPINGS -------------------- #
mapping_type = {
    'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,
    'Random':5,'Scratch':6,'none':7,'[0 0]':7,'Unknown':7
}
inv_mapping = {v:k for k,v in mapping_type.items()}

def map_label(label):
    try:
        label_int = int(label)
        return inv_mapping.get(label_int, f"Unknown ({label})")
    except:
        return str(label)

def map_wafer_to_rgb(wafer_map):
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10,10,3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H, W = wafer_map.shape
    rgb = 50 * np.ones((H, W, 3), dtype=np.uint8)
    rgb[wafer_map==1] = [0,255,255]   # Functional die
    rgb[wafer_map==2] = [255,0,0]     # Defective die
    return rgb

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32,32)):
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)
    if isinstance(wafer_map, Image.Image):
        wafer_resized = wafer_map.convert('L').resize(target_dim, Image.NEAREST)
    elif wafer_map.ndim == 2:
        wafer_resized = Image.fromarray(wafer_map.astype(np.uint8)).resize(target_dim, Image.NEAREST)
    else:
        return np.zeros(required_size)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size:
        return np.pad(X_flat,(0,required_size-current_size),'constant')
    else:
        return X_flat[:required_size]

# -------------------- STREAMLIT TABS -------------------- #
tabs = st.tabs(["Predict Defects","Model Insights","About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    # --- CNN MODEL --- #
    if model_choice == "CNN (Image-Based)" and cnn_pipe:
        st.subheader("Upload wafer images (.npy or .png/.jpg/.jpeg)")
        uploaded_files = st.file_uploader("Upload wafer maps", type=["png","jpg","jpeg","npy"], accept_multiple_files=True)
        
        if uploaded_files:
            results = []
            for file in uploaded_files:
                try:
                    if file.name.endswith(".npy"):
                        wafer_data = np.load(file)
                        img = Image.fromarray(wafer_data.astype(np.uint8))
                    else:
                        img = Image.open(file).convert("L")
                        wafer_data = np.array(img)
                    label, probs = cnn_pipe.predict(img)
                    results.append({"File": file.name, "Predicted_Label": label, "Probabilities": probs, "Wafer_Data": wafer_data})
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Display predictions
        if st.session_state.cnn_results:
            idx = st.session_state.cnn_index
            if len(st.session_state.cnn_results) > 1:
                idx = st.slider("Select image", 0, len(st.session_state.cnn_results)-1, idx)
                st.session_state.cnn_index = idx
            r = st.session_state.cnn_results[idx]
            st.image(map_wafer_to_rgb(r['Wafer_Data']), width=200, caption=r['File'])
            st.markdown(f"**Predicted:** {r['Predicted_Label']}")
            if isinstance(r['Probabilities'], dict):
                for label, prob in sorted(r['Probabilities'].items(), key=lambda x:x[1], reverse=True):
                    st.progress(np.clip(prob,0,1))
                    st.markdown(f"**{label}**: {prob:.2f}")

    # --- XGBOOST MODEL --- #
    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.subheader("Upload wafer images or .npy feature arrays")
        uploaded_files = st.file_uploader("Upload wafer maps", type=["npy","png","jpg","jpeg"], accept_multiple_files=True)
        if uploaded_files:
            results = []
            for file in uploaded_files:
                try:
                    wafer_input = np.load(file) if file.name.endswith(".npy") else Image.open(file).convert("L")
                    X_feat = prepare_pixel_features_for_xgb(wafer_input).reshape(1,-1)
                    X_scaled = scaler.transform(X_feat)
                    pred_idx = int(xgb.predict(X_scaled)[0])
                    pred_label = inv_mapping.get(pred_idx, f"Unknown ({pred_idx})")
                    results.append({"File": file.name, "Predicted_Label": pred_label})
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
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
        st.markdown(f"### Analysis for Wafer: **{r['File']}**")
        st.markdown(f"**Primary Prediction:** <span style='color:#ff0000'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
        if isinstance(r['Probabilities'], dict):
            prob_df = pd.DataFrame({'Defect Type': list(r['Probabilities'].keys()), 
                                    'Confidence': r['Probabilities'].values()}).sort_values(by='Confidence', ascending=False)
            st.subheader("Confidence Distribution")
            st.bar_chart(prob_df.set_index('Defect Type'), height=350)
    else:
        st.warning("Please run a prediction first.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps
    - **XGBoost** for feature-based wafer data
    - **Focal Loss** to handle class imbalance
    - **Streamlit** for interactive dashboard
    """)
    st.image("https://placehold.co/600x400/1e293b/f8fafc?text=Example+Wafer+Defect+Map", width=300)