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

# --- NEW FEATURE EXTRACTION FUNCTION FOR XGBOOST ---
def extract_features_from_wafer(wafer_map):
    """
    Extracts 10 numerical features from the 2D wafer map for the XGBoost model.
    Assumes the wafer map is a 2D array of 0, 1, 2 values.
    """
    if wafer_map is None or wafer_map.size == 0:
        return np.zeros(10)

    # 1. Total Die Count (Value 1 + Value 2)
    total_die = np.sum(wafer_map > 0)
    
    # Check if there are any die to prevent division by zero
    if total_die == 0:
        return np.zeros(10)

    # 2. Defect Die Count (Value 2)
    defect_die = np.sum(wafer_map == 2)

    # 3. Defect Percentage
    defect_percent = defect_die / total_die

    # Isolate functional and defect areas (mask)
    functional_map = (wafer_map == 1).astype(np.float32)
    defect_map = (wafer_map == 2).astype(np.float32)
    
    # 4. Mean of Functional Area
    mean_func = np.mean(functional_map)

    # 5. Standard Deviation of Functional Area
    std_func = np.std(functional_map)

    # 6. Mean of Defect Area (Only meaningful if defects exist)
    mean_defect = np.mean(defect_map)

    # 7. Aspect Ratio of Wafer Bounding Box (Approximation)
    # Find bounding box coordinates (where wafer_map > 0)
    rows, cols = np.where(wafer_map > 0)
    if len(rows) > 1:
        height = np.max(rows) - np.min(rows) + 1
        width = np.max(cols) - np.min(cols) + 1
        aspect_ratio = width / height if height > 0 else 0
    else:
        aspect_ratio = 0

    # 8. Density/Coverage (Ratio of die area to total area)
    coverage = total_die / wafer_map.size
    
    # 9. Center of Mass of Defects (Row coordinate, normalized)
    # Using scipy.ndimage.center_of_mass
    if defect_die > 0:
        center_of_mass = ndimage.center_of_mass(defect_map)
        normalized_row_center = center_of_mass[0] / wafer_map.shape[0] if wafer_map.shape[0] > 0 else 0
    else:
        normalized_row_center = 0

    # 10. Simple Image Moment (M00 of defects) - essentially the defect area
    moment_m00 = defect_die 

    # Bundle the 10 features
    features = np.array([
        total_die, defect_die, defect_percent, mean_func, std_func,
        mean_defect, aspect_ratio, coverage, normalized_row_center, moment_m00
    ])

    return features

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
        st.subheader("Upload wafer image/feature arrays to predict (Requires 10 features)")
        uploaded_files = st.file_uploader(
            # --- UPDATED: Allow image formats here ---
            "Upload feature arrays (.npy) or images (.png, .jpg, .jpeg)", 
            type=["npy", "png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            results = []
            for file in uploaded_files:
                
                # --- XGBOOST PROCESSING LOGIC ---
                if file.name.endswith(".npy"):
                    # Case 1: Raw feature array (10 features) uploaded
                    X = np.load(file).reshape(1,-1)
                    
                else:
                    # Case 2: Image file uploaded -> Must extract features
                    try:
                        img = Image.open(file).convert("L")
                        wafer = np.array(img)
                        
                        # Use the new extraction function to get the 10 features
                        X = extract_features_from_wafer(wafer).reshape(1,-1)
                        
                    except Exception as e:
                        st.error(f"Error processing image {file.name} for XGBoost feature extraction: {e}")
                        continue
                
                # --- Prediction and Scaling ---
                try:
                    # Check if the feature vector size is correct before scaling
                    if X.shape[1] != 10: 
                        st.error(f"Feature count mismatch for {file.name}. Expected 10 features, got {X.shape[1]}.")
                        continue

                    X_scaled = scaler.transform(X)
                    pred = xgb.predict(X_scaled)[0]
                    results.append({"File": file.name, "Predicted_Label": map_label(str(pred))})
                except ValueError as e:
                    st.error(f"Model/Scaler error for {file.name}: {e}")
            
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
