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

# -------------------- LOAD MODELS -------------------- #
cnn_pipe = None
if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.warning("CNN model or label encoder not found.")

xgb, scaler = None, None
if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
    xgb = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.warning("XGBoost model or scaler not found.")

# -------------------- DEFECT LABEL MAPPING -------------------- #
mapping_type = {
    'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,
    'Loc':4,'Random':5,'Scratch':6,'none':7,'[0 0]':7,'Unknown':7
}
# Reverse mapping for display
inv_mapping_type = {v:k for k,v in mapping_type.items()}

def map_label(label):
    try:
        label_int = int(label)
    except:
        return str(label)
    return inv_mapping_type.get(label_int, f"Unknown ({label_int})")

# -------------------- UTILITY FUNCTIONS -------------------- #
def map_wafer_to_rgb(wafer_map):
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10,10,3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H,W = wafer_map.shape
    rgb_image = 50 * np.ones((H,W,3), dtype=np.uint8)
    rgb_image[wafer_map==1] = [0,255,255]  # Functional Die
    rgb_image[wafer_map==2] = [255,0,0]    # Defective Die
    return rgb_image

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32,32)):
    if wafer_map is None or wafer_map.size==0:
        return np.zeros(required_size)
    if wafer_map.ndim !=2:
        if wafer_map.ndim>2 and wafer_map.shape[2]==3:
            wafer_map = np.array(Image.fromarray(wafer_map,'RGB').convert('L'))
        elif wafer_map.size==target_dim[0]*target_dim[1]:
            wafer_map = wafer_map.reshape(target_dim)
        else:
            return np.zeros(required_size)
    wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size:
        return np.pad(X_flat, (0, required_size-current_size), 'constant')
    else:
        return X_flat[:required_size]

# -------------------- STREAMLIT TABS -------------------- #
tabs = st.tabs(["Predict Defects","Model Insights","About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])
    
    uploaded_files = st.file_uploader(
        "Upload wafer maps (.png, .jpg, .jpeg, .npy)", 
        type=["png","jpg","jpeg","npy"], accept_multiple_files=True
    )
    
    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            wafer = None
            if uploaded_file.name.endswith(".npy"):
                wafer = np.load(uploaded_file)
            else:
                wafer = np.array(Image.open(uploaded_file).convert("L"))
            
            if model_choice=="CNN (Image-Based)" and cnn_pipe:
                label, probs = cnn_pipe.predict(wafer)
                results.append({"File":uploaded_file.name,"Predicted_Label":label,"Probabilities":probs,"Wafer_Data":wafer})
            elif model_choice=="XGBoost (Feature-Based)" and xgb:
                X = prepare_pixel_features_for_xgb(wafer).reshape(1,-1)
                X_scaled = scaler.transform(X)
                pred_int = int(xgb.predict(X_scaled)[0])
                results.append({"File":uploaded_file.name,"Predicted_Label":map_label(pred_int)})
        
        if model_choice=="CNN (Image-Based)":
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0
        else:
            st.session_state.xgb_results = results
        
    # Display predictions
    if model_choice=="CNN (Image-Based)" and st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        wafer_rgb_display = map_wafer_to_rgb(r['Wafer_Data'])
        st.image(wafer_rgb_display, width=200, caption=f"Wafer Map: {r['File']}")
        st.markdown(f"**Predicted:** {map_label(r['Predicted_Label'])}")
        probs = r['Probabilities']
        if isinstance(probs, dict):
            st.subheader("Top Probabilities")
            for k,v in sorted(probs.items(), key=lambda x:x[1], reverse=True)[:3]:
                st.markdown(f"**{map_label(k)}:** {v:.2f}")
        col1,col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.cnn_index = max(0,st.session_state.cnn_index-1)
        with col2:
            if st.button("Next"):
                st.session_state.cnn_index = min(len(st.session_state.cnn_results)-1,st.session_state.cnn_index+1)

    elif model_choice=="XGBoost (Feature-Based)" and st.session_state.xgb_results:
        st.subheader("XGBoost Predictions:")
        for r in st.session_state.xgb_results:
            st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights for Current Prediction")
    if st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        st.markdown(f"### Wafer: {r['File']}")
        st.markdown(f"**Predicted:** {map_label(r['Predicted_Label'])}")
        probs = r['Probabilities']
        if isinstance(probs, dict):
            prob_df = pd.DataFrame({'Defect Type':[map_label(k) for k in probs.keys()],
                                    'Confidence':probs.values()}).sort_values('Confidence',ascending=False)
            st.bar_chart(prob_df.set_index('Defect Type'), height=300)
    else:
        st.warning("Please upload or predict a wafer first to view insights.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    ChipSleuth detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps
    - **XGBoost** for feature-based wafer data
    - **Streamlit** for interactive dashboard
    """)
    st.image("https://placehold.co/600x600/1e293b/f8fafc?text=Example+Wafer+Map", width=300)