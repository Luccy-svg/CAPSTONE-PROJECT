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
if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.warning("CNN model or label encoder not found.")
    cnn_pipe = None

if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
    xgb = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.warning("XGBoost model or scaler not found.")
    xgb = None
    scaler = None

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
    rgb[wafer_map==1] = [0,255,255]
    rgb[wafer_map==2] = [255,0,0]
    return rgb

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32,32)):
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)
    if wafer_map.ndim != 2:
        if wafer_map.ndim==3 and wafer_map.shape[2]==3:
            wafer_map = np.array(Image.fromarray(wafer_map,'RGB').convert('L'))
        else:
            return np.zeros(required_size)
    wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size:
        return np.pad(X_flat,(0,required_size-current_size),'constant')
    else:
        return X_flat[:required_size]

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects","Model Insights","About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    # --- CNN MODEL --- #
    if model_choice == "CNN (Image-Based)" and cnn_pipe:
        st.subheader("Upload wafer images (.npy or .png/.jpg/.jpeg)")
        uploaded_files = st.file_uploader(
            "Upload wafer maps", 
            type=["png","jpg","jpeg","npy"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer = None
                if file.name.endswith(".npy"):
                    wafer = np.load(file)
                else:
                    img = Image.open(file).convert("L")
                    wafer = np.array(img)
                label, probs = cnn_pipe.predict(wafer)
                results.append({
                    "File": file.name,
                    "Predicted_Label": label,
                    "Probabilities": probs,
                    "Wafer_Data": wafer
                })
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Display current wafer and probabilities with slider
        if st.session_state.cnn_results:
            idx = st.slider(
                "Select image to view",
                min_value=0,
                max_value=len(st.session_state.cnn_results)-1,
                value=st.session_state.cnn_index,
                key="cnn_slider"
            )
            st.session_state.cnn_index = idx

            r = st.session_state.cnn_results[idx]
            wafer_rgb = map_wafer_to_rgb(r['Wafer_Data'])
            st.image(wafer_rgb, width=200, caption=f"Wafer Map: {r['File']}")
            st.markdown(f"**Predicted:** {r['Predicted_Label']}")
            st.subheader("Prediction Probabilities")
            probs = r['Probabilities']
            if isinstance(probs, dict):
                top_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
                for label, prob in top_probs:
                    st.progress(np.clip(prob,0,1))
                    st.markdown(f"**{label}**: {prob:.2f}")

    # --- XGBOOST MODEL --- #
    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.subheader("Upload wafer images or .npy feature arrays")
        uploaded_files = st.file_uploader(
            "Upload wafer maps", 
            type=["npy","png","jpg","jpeg"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer = None
                if file.name.endswith(".npy"):
                    wafer = np.load(file)
                else:
                    img = Image.open(file).convert("L")
                    wafer = np.array(img)
                X_feat = prepare_pixel_features_for_xgb(wafer).reshape(1,-1)
                X_scaled = scaler.transform(X_feat)
                pred_idx = int(xgb.predict(X_scaled)[0])
                pred_label = inv_mapping.get(pred_idx,f"Unknown ({pred_idx})")
                results.append({"File":file.name,"Predicted_Label":pred_label,"Raw_Pred":pred_idx})
            st.session_state.xgb_results = results
            st.subheader("XGBoost Predictions (mapped to defect types):")
            for r in results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights for Current Prediction")
    if st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        st.markdown(f"### Analysis for Wafer: **{r['File']}**")
        st.markdown(f"**Primary Prediction:** <span style='color:#00ffff'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
        if isinstance(r['Probabilities'], dict):
            prob_df = pd.DataFrame({
                'Defect Type': [map_label(k) for k in r['Probabilities'].keys()],
                'Confidence': r['Probabilities'].values()
            }).sort_values(by='Confidence', ascending=False)
            st.subheader("Confidence Distribution Across Defect Classes")
            st.bar_chart(prob_df.set_index('Defect Type'), height=350)
    else:
        st.warning("Please run a prediction on the 'Predict Defects' tab first.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps
    - **XGBoost** for feature-based wafer data
    - **Streamlit** for interactive dashboard deployment
    """)
    st.image("https://placehold.co/600x600/1e293b/f8fafc?text=Example+Wafer+Map", width=300)