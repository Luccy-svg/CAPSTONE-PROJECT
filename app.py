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
import pandas as pd
# Assuming cnn_pipeline is available and contains WaferCNNPipeline
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
# NEW STATE: Store extracted features for display
if "xgb_features" not in st.session_state:
    st.session_state.xgb_features = {}

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

# Standard Defect ID to Label Mapping for Wafer Inspection datasets
DEFECT_MAP = {
    0: 'No Defect',
    1: 'Center',
    2: 'Donut',
    3: 'Edge-Ring',
    4: 'Scratch',
    5: 'Near-full',
    6: 'Random',
    7: 'Local (Loc)', 
    8: 'Cluster',
    '0 0': 'No Defect' # Used by CNN pipeline for no defect
}

def map_label(label):
    """
    Maps numerical or '0 0' labels to human-readable defect names using DEFECT_MAP.
    """
    # Ensure label is int if possible for lookup
    try:
        label_key = int(label)
    except ValueError:
        label_key = str(label)
        
    return DEFECT_MAP.get(label_key, f"Unknown Defect ID: {label}")

def map_wafer_to_rgb(wafer_map):
    """
    Maps discrete wafer map values (0, 1, 2) to distinct, high-contrast RGB colors 
    for optimal visualization in the dashboard.
    """
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10, 10, 3), dtype=np.uint8)

    wafer_map = wafer_map.astype(np.int8)

    H, W = wafer_map.shape
    rgb_image = 50 * np.ones((H, W, 3), dtype=np.uint8) # Dark Gray Background (0)

    rgb_image[wafer_map == 1] = [0, 255, 255] # Functional Die -> Bright Cyan
    rgb_image[wafer_map == 2] = [255, 0, 0] # Defect Die -> Bright Red
    
    return rgb_image

# --- PIXEL PREPARATION FUNCTION (UPDATED FOR 32x32 RESIZE) ---
def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32, 32)):
    """
    Prepares the wafer map for the XGBoost model by resizing, flattening, and padding 
    to exactly match the expected feature count (1029) indicated by the loaded scaler.
    
    Based on the training script (X_combined), this assumes the input is:
    (32x32 pixels flattened) + 5 additional features = 1029 features.
    """
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)

    # 1. Ensure map is 2D and prepare for resizing
    if wafer_map.ndim != 2:
        if wafer_map.ndim == 3 and wafer_map.shape[2] == 3:
            # Convert color image to grayscale/monochrome
            wafer_map = Image.fromarray(wafer_map, 'RGB').convert('L')
            wafer_map = np.array(wafer_map)
        else:
            if wafer_map.ndim > 2 and wafer_map.size == target_dim[0] * target_dim[1]:
                 wafer_map = wafer_map.reshape(target_dim)
            else:
                 return np.zeros(required_size) 

    # 2. Resize the wafer map to the target dimension (32x32)
    wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
    # Use NEAREST neighbor resampling for categorical data (0, 1, 2)
    wafer_resized = wafer_img.resize(target_dim, Image.NEAREST) 
    wafer_array = np.array(wafer_resized)
    
    # 3. Flatten the array (32 * 32 = 1024 features)
    X_flat = wafer_array.flatten()
    current_size = len(X_flat)

    # 4. Pad or Truncate to match the required 1029 features
    if current_size < required_size:
        # Pad with 5 zeros to meet the required 1029 features (1024 + 5 = 1029)
        padding_needed = required_size - current_size
        return np.pad(X_flat, (0, padding_needed), 'constant', constant_values=0)
    else:
        # Truncate if larger (should not happen with 32x32 resize)
        return X_flat[:required_size]


# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    # -------------------- CNN INTERACTIVE -------------------- #
    if model_choice == "CNN (Image-Based)":
        st.subheader("Upload wafer images (.npy or .png) to run prediction")
        uploaded_files = st.file_uploader(
            "Upload wafer maps (.png, .jpg, .jpeg, .npy)", 
            type=["png","jpg","jpeg","npy"], accept_multiple_files=True
        )

        # Predict uploaded
        if uploaded_files and cnn_pipe:
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

        # Display interactive image
        if st.session_state.cnn_results:
            idx = st.session_state.cnn_index
            r = st.session_state.cnn_results[idx]
            wafer = r.get("Wafer_Data") 

            if wafer is not None:
                wafer_rgb_display = map_wafer_to_rgb(wafer)
                st.image(wafer_rgb_display, width=200, caption=f"Wafer Map: {r['File']}") 
                
            st.markdown(f"**Predicted:** {map_label(r['Predicted_Label'])}")
            
            # Probability Distribution (Optional Insight) 
            probs = r['Probabilities']
            if isinstance(probs, dict):
                 top_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
                 st.caption("Top Predictions:")
                 for label, prob in top_probs:
                    progress_value = np.clip(prob, 0.0, 1.0)
                    st.progress(progress_value)
                    st.markdown(f"**{map_label(label)}**: {prob:.2f}")

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
        # --- WARNING ABOUT FEATURE MISMATCH ---
        st.warning("""
        ⚠️ **XGBoost Model Input Synchronization:** The app now enforces resizing of all uploaded images to **32x32 pixels** and pads them to **1029 features** to match the training input of the loaded model and scaler.
        """)
        st.subheader("Upload wafer image/feature arrays to predict (Input is resized to 32x32)")
        uploaded_files = st.file_uploader(
            "Upload feature arrays (.npy) or images (.png, .jpg, .jpeg)", 
            type=["npy", "png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            results = []
            for file in uploaded_files:
                
                # --- XGBOOST PROCESSING LOGIC ---
                X = None
                wafer = None
                
                if file.name.endswith(".npy"):
                    # Case 1: Raw NumPy array uploaded. 
                    try:
                        wafer = np.load(file)
                    except Exception as e:
                        st.error(f"Error loading NumPy file {file.name}: {e}")
                        continue
                else:
                    # Case 2: Image file uploaded 
                    try:
                        img = Image.open(file).convert("L")
                        wafer = np.array(img)
                    except Exception as e:
                        st.error(f"Error reading image {file.name} for XGBoost: {e}")
                        continue
                
                # Use the new preparation function to get the 1029 features
                if wafer is not None:
                    try:
                        X = prepare_pixel_features_for_xgb(wafer).reshape(1, -1)
                    except Exception as e:
                        st.error(f"Error preparing pixel features for {file.name}: {e}")
                        continue
                
                # --- Prediction and Scaling ---
                try:
                    # Explicitly check for the required feature size: 1029
                    required_features = 1029 
                    if X is None or X.shape[1] != required_features: 
                        st.error(f"Feature count mismatch for {file.name}. Expected {required_features} features, got {X.shape[1] if X is not None else 0}.")
                        continue
                    
                    # Ensure integer prediction for correct label mapping
                    X_scaled = scaler.transform(X)
                    pred = int(xgb.predict(X_scaled)[0]) 
                    results.append({"File": file.name, "Predicted_Label": map_label(pred)})
                except ValueError as e:
                    st.error(f"Model/Scaler error for {file.name}: {e}")
            
            st.session_state.xgb_results = results
            st.subheader("XGBoost Predictions (Input Resized to 32x32 Pixels):")
            for r in results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights for Current Prediction")
    
    if st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        current_file = r['File']
        predicted_label = map_label(r['Predicted_Label'])
        probs = r['Probabilities']
        
        st.markdown(f"### Analysis for Wafer: **{current_file}**")
        st.markdown(f"**Primary Prediction:** <span style='color: #00ffff; font-size: 1.2em;'>**{predicted_label}**</span>", unsafe_allow_html=True)

        if isinstance(probs, dict):
            
            # --- Probability Distribution (Heat Map Equivalent) ---
            st.subheader("Confidence Distribution Across Defect Classes")
            st.write("This chart visualizes the model's confidence for each possible defect type for the current wafer.")
            
            # Prepare data for charting
            prob_df = pd.DataFrame(
                {'Defect Type': [map_label(k) for k in probs.keys()], 'Confidence': probs.values()}
            )
            prob_df = prob_df.sort_values(by='Confidence', ascending=False)
            
            # Use Streamlit bar chart for clear probability visualization
            st.bar_chart(prob_df.set_index('Defect Type'), height=350)
            
            
            # --- Conceptual Confusion Matrix Section ---
            st.subheader("Conceptual Confusion Matrix Analysis")
            
            st.write("""
            Since we are analyzing single predictions without a known ground truth label, we can't show a full confusion matrix 
            (True Positives, False Negatives, etc.). 
            """)
            
            # Display a simplified prediction table
            cm_data = {
                'Metric': ['Predicted Type', 'Model Confidence'],
                'Value': [predicted_label, f"{probs[r['Predicted_Label']]:.4f}"]
            }
            st.table(pd.DataFrame(cm_data))

            st.info("""
            **Full Confusion Matrix:** To view a full confusion matrix, the model would need to be run against a pre-labeled test set to calculate overall performance metrics.
            """)

    else:
        st.warning("Please upload a wafer image or run a prediction on the 'Predict Defects' tab first to view insights.")

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
    # Illustrate a wafer defect map since the whole app revolves around it
    st.image("https://placehold.co/600x600/1e293b/f8fafc?text=Example+Wafer+Map+with+Defects", caption="Conceptual Wafer Map showing defect patterns (e.g., 'Donut', 'Scratch').", width=300)
