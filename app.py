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
# Use st.cache_resource for heavy model loading to speed up app reloading
@st.cache_resource
def load_cnn_pipeline():
    if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        try:
            return WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
        except Exception as e:
            # Added a better fallback error message based on the custom loss function
            st.error(f"Error loading CNN Pipeline: {e}. Check if the custom Focal Loss is correctly defined in cnn_pipeline.py.")
            return None
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
            st.error(f"Error loading XGBoost models: {e}.")
            return None, None
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
    # Ensure wafer map is integer type for indexing
    wafer_map = wafer_map.astype(np.int8) 
    H, W = wafer_map.shape
    # Default to gray background (non-functional area)
    rgb = 50 * np.ones((H, W, 3), dtype=np.uint8) 
    # Functional die (value 1) - Cyan/Teal
    rgb[wafer_map==1] = [0,255,255] 
    # Defective die (value 2) - Red
    rgb[wafer_map==2] = [255,0,0] 
    return rgb

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32,32)):
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(required_size)
    
    # Handle PIL Image input (from the updated Streamlit loop)
    if isinstance(wafer_map, Image.Image):
        # Use Image.NEAREST for proper resizing of categorical data
        wafer_resized = wafer_map.convert('L').resize(target_dim, Image.NEAREST)
        
    elif wafer_map.ndim == 2:
        # NumPy array case (e.g., loaded from .npy)
        wafer_img = Image.fromarray(wafer_map.astype(np.uint8))
        wafer_resized = wafer_img.resize(target_dim, Image.NEAREST)
        
    else:
        # Fallback for unexpected array shapes
        return np.zeros(required_size)
            
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
        uploaded_files = st.file_uploader("Upload wafer maps", type=["png","jpg","jpeg","npy"], accept_multiple_files=True)
        
        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer_data_for_display = None
                wafer_for_cnn_predict = None # Initialize variable for pipeline input

                # File handling and type conversion
                if file.name.endswith(".npy"):
                    wafer_data_for_display = np.load(file)
                    # For CNN pipeline, convert NumPy array back to PIL image
                    wafer_for_cnn_predict = Image.fromarray(wafer_data_for_display.astype(np.uint8))
                else:
                    # Load as PIL Image (required by pipeline)
                    img = Image.open(file).convert("L")
                    wafer_for_cnn_predict = img
                    # Create NumPy array for display/storage
                    wafer_data_for_display = np.array(img)
                
                # CRITICAL FIX: Pass the PIL Image object
                if wafer_for_cnn_predict:
                    try:
                        label, probs = cnn_pipe.predict(wafer_for_cnn_predict)
                        results.append({
                            "File":file.name,
                            "Predicted_Label":label,
                            "Probabilities":probs,
                            "Wafer_Data":wafer_data_for_display # NumPy array for display
                        })
                    except ValueError as e:
                         st.error(f"Prediction failed for {file.name}. Ensure image format is correct. Error: {e}")


            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Display current wafer and probabilities
        if st.session_state.cnn_results:
            num_results = len(st.session_state.cnn_results)
            
            # --- START SLIDER READABILITY FIX ---
            if num_results > 1:
                # User-friendly display: 1 to N
                display_value = st.session_state.cnn_index + 1 
                
                # Slider input is based on 1 to N
                idx_display = st.slider(
                    f"Select image to view (File 1 of {num_results} to File {num_results} of {num_results})",
                    min_value=1,                      # Start at 1
                    max_value=num_results,            # End at total count
                    value=display_value,              # Current value is 1-based
                    key="cnn_slider"
                )
                # Store the 0-based index internally
                st.session_state.cnn_index = idx_display - 1 
                idx = st.session_state.cnn_index
            else:
                idx = 0
                st.session_state.cnn_index = 0
            # --- END SLIDER READABILITY FIX ---

            r = st.session_state.cnn_results[idx]
            # Wafer_Data is a NumPy array here, which is what map_wafer_to_rgb expects
            wafer_rgb = map_wafer_to_rgb(r['Wafer_Data'])
            
            # Show the file number in the caption for better tracking
            st.image(
                wafer_rgb, 
                width=200, 
                caption=f"File {idx + 1} of {num_results}: {r['File']}"
            )
            
            st.markdown(f"**Predicted:** <span style='color:#ff0000; font-size: 1.5em; font-weight: bold;'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
            
            st.subheader("Prediction Probabilities")
            
            probs = r['Probabilities']
            if isinstance(probs, dict):
                top_probs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
                for label, prob in top_probs:
                    # Clip probability to [0, 1] for progress bar
                    st.progress(np.clip(prob, 0, 1)) 
                    # Use markdown for consistent display
                    st.markdown(f"**{label}**: {prob:.2f}")

    # --- XGBOOST MODEL --- #
    elif model_choice == "XGBoost (Feature-Based)" and xgb:
        st.subheader("Upload wafer images or .npy feature arrays")
        uploaded_files = st.file_uploader("Upload wafer maps", type=["npy","png","jpg","jpeg"], accept_multiple_files=True)
        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer_input = None
                if file.name.endswith(".npy"):
                    wafer_input = np.load(file)
                else:
                    # Use PIL Image for consistent feature extraction input
                    wafer_input = Image.open(file).convert("L") 
                
                # Feature extraction and scaling
                X_feat = prepare_pixel_features_for_xgb(wafer_input).reshape(1,-1)
                X_scaled = scaler.transform(X_feat)
                
                # Prediction
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
        st.markdown(f"**Primary Prediction:** <span style='color:#ff0000; font-size: 1.5em; font-weight: bold;'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
        if isinstance(r['Probabilities'], dict):
            prob_df = pd.DataFrame({
                # Ensure the keys are converted to strings before creating the DataFrame
                'Defect Type': list(r['Probabilities'].keys()), 
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
    - **CNN** for image-based wafer maps (leveraging image patterns)
    - **XGBoost** for feature-based wafer data (leveraging pixel statistics)
    - **Streamlit** for interactive dashboard deployment
    
    The **Focal Loss** function was used to train the CNN, which helps the model focus on hard-to-classify and less common defect types (like **Scratch** or **Cluster**), improving overall classification accuracy across all classes.
    """)
    st.image("https://placehold.co/600x400/1e293b/f8fafc?text=Example+Wafer+Defect+Map", width=300)
