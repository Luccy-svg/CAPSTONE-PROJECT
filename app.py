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
for key in ["cnn_results", "cnn_index", "xgb_results", "xgb_features"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "results" in key else 0 if "index" in key else {}

# -------------------- PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"  # Use best model
LABEL_ENCODER_PATH = "label_encoder.pkl"
XGB_MODEL_PATH = "xgboost_improved.pkl"
SCALER_PATH = "scaler.pkl"

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
    st.warning("XGBoost model or scaler not found.")
    xgb, scaler = None, None

# -------------------- UTILITY FUNCTIONS -------------------- #
DEFECT_MAP = {
    0: 'No Defect', 1: 'Center', 2: 'Donut', 3: 'Edge-Ring',
    4: 'Scratch', 5: 'Near-full', 6: 'Random', 7: 'Local (Loc)', 8: 'Cluster',
    '0 0': 'No Defect'
}

def map_label(label):
    try:
        label_key = int(label)
    except ValueError:
        label_key = str(label)
    return DEFECT_MAP.get(label_key, f"Unknown Defect ID: {label}")

def map_wafer_to_rgb(wafer_map):
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10,10,3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H, W = wafer_map.shape
    rgb_image = 50 * np.ones((H, W, 3), dtype=np.uint8)
    rgb_image[wafer_map == 1] = [0,255,255]  # functional
    rgb_image[wafer_map == 2] = [255,0,0]    # defect
    return rgb_image

def prepare_wafer_for_cnn(wafer, target_size=(32,32)):
    """Resize, add channel, normalize to [0,1]"""
    if wafer.ndim > 2:
        wafer = wafer[...,0]
    img = Image.fromarray(wafer.astype(np.uint8)).resize(target_size, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)/255.0
    arr = np.expand_dims(arr, axis=-1)  # channel
    arr = np.expand_dims(arr, axis=0)   # batch
    return arr

def prepare_pixel_features_for_xgb(wafer, required_size=1029, target_dim=(32,32)):
    """Flatten wafer and pad/truncate to required features for XGBoost"""
    if wafer is None or wafer.size==0:
        return np.zeros(required_size)
    if wafer.ndim !=2:
        if wafer.ndim==3 and wafer.shape[2]==3:
            wafer = np.array(Image.fromarray(wafer,'RGB').convert('L'))
        elif wafer.ndim>2 and wafer.size==target_dim[0]*target_dim[1]:
            wafer = wafer.reshape(target_dim)
        else:
            return np.zeros(required_size)
    wafer_img = Image.fromarray(wafer.astype(np.uint8))
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
    X_flat = np.array(wafer_resized).flatten()
    pad_len = required_size - len(X_flat)
    if pad_len>0:
        return np.pad(X_flat,(0,pad_len),'constant')
    return X_flat[:required_size]

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])
    
    uploaded_files = st.file_uploader(
        "Upload wafer images (.png, .jpg, .jpeg, .npy)", 
        type=["png","jpg","jpeg","npy"], accept_multiple_files=True
    )
    
    # --- CNN PREDICTION ---
    if model_choice=="CNN (Image-Based)" and cnn_pipe and uploaded_files:
        results=[]
        for f in uploaded_files:
            wafer = np.load(f) if f.name.endswith(".npy") else np.array(Image.open(f).convert("L"))
            X_cnn = prepare_wafer_for_cnn(wafer)
            probs_arr = cnn_pipe.model.predict(X_cnn, verbose=0)[0]
            pred_idx = np.argmax(probs_arr)
            pred_label = int(cnn_pipe.le.inverse_transform([pred_idx])[0])
            probs_dict = {int(cnn_pipe.le.inverse_transform([i])[0]): float(p) for i,p in enumerate(probs_arr)}
            results.append({"File":f.name,"Predicted_Label":pred_label,"Probabilities":probs_dict,"Wafer_Data":wafer})
        st.session_state.cnn_results=results
        st.session_state.cnn_index=0
        
        # Display first
        r=st.session_state.cnn_results[0]
        st.image(map_wafer_to_rgb(r["Wafer_Data"]), width=200, caption=f"Wafer Map: {r['File']}")
        st.markdown(f"**Predicted:** {map_label(r['Predicted_Label'])}")
    
    # --- XGBoost PREDICTION ---
    elif model_choice=="XGBoost (Feature-Based)" and xgb and uploaded_files:
        results=[]
        for f in uploaded_files:
            wafer = np.load(f) if f.name.endswith(".npy") else np.array(Image.open(f).convert("L"))
            X_feat = prepare_pixel_features_for_xgb(wafer).reshape(1,-1)
            if X_feat.shape[1] != xgb.n_features_in_:
                st.warning(f"Feature mismatch: expected {xgb.n_features_in_}, got {X_feat.shape[1]}")
                continue
            X_scaled = scaler.transform(X_feat)
            pred = int(xgb.predict(X_scaled)[0])
            results.append({"File":f.name,"Predicted_Label":map_label(pred)})
        st.session_state.xgb_results=results
        for r in results:
            st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights for Current Prediction")
    if st.session_state.cnn_results:
        idx=st.session_state.cnn_index
        r=st.session_state.cnn_results[idx]
        st.markdown(f"### Wafer: **{r['File']}**")
        st.markdown(f"**Predicted Type:** {map_label(r['Predicted_Label'])}")
        probs=r['Probabilities']
        prob_df=pd.DataFrame({'Defect Type':[map_label(k) for k in probs.keys()],
                              'Confidence':probs.values()}).sort_values(by='Confidence',ascending=False)
        st.bar_chart(prob_df.set_index('Defect Type'),height=350)
    else:
        st.warning("Upload and predict a wafer first to see insights.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    Detect semiconductor wafer defects using CNN (image) & XGBoost (features) with Streamlit.
    """)