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

# -------------------- CNN PIPELINE -------------------- #
from tensorflow.keras.models import load_model

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=1)
    return focal_loss_fixed

class WaferCNNPipeline:
    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32,32)):
        self.model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()}, compile=False)
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image: Image.Image) -> np.ndarray:
        if not isinstance(wafer_image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
        wafer_array = np.array(wafer_image, dtype=np.float32)

        # --- FIX: Robust scaling ---
        if wafer_array.max() > 2:
            wafer_array = wafer_array / 127.5  # scale 0-255 -> 0-2
            wafer_array = np.clip(np.round(wafer_array), 0, 2)

        wafer_array = wafer_array.astype(np.float32)
        wafer_array = wafer_array.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_array

    def predict(self, wafer_image: Image.Image):
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        label = self.le.inverse_transform([pred_class])[0] if pred_class < len(self.le.classes_) else f"Unknown ({pred_class})"
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) for i in range(len(self.le.classes_))}
        return label, probs

# -------------------- STREAMLIT CONFIG -------------------- #
st.set_page_config(page_title="ChipSleuth: Wafer Defect Dashboard", page_icon="🕵️‍♀️", layout="wide")
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
            st.error(f"Error loading CNN Pipeline: {e}")
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
            st.error(f"Error loading XGBoost models: {e}")
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

def map_wafer_to_rgb(wafer_map):
    if wafer_map is None or wafer_map.size == 0:
        return 50 * np.ones((10,10,3), dtype=np.uint8)
    wafer_map = wafer_map.astype(np.int8)
    H, W = wafer_map.shape
    rgb = 50 * np.ones((H,W,3), dtype=np.uint8)
    rgb[wafer_map==1] = [0,255,255]
    rgb[wafer_map==2] = [255,0,0]
    return rgb

def prepare_pixel_features_for_xgb(wafer_map, required_size=1029, target_dim=(32,32)):
    if wafer_map is None or wafer_map.size == 0: return np.zeros(required_size)
    if isinstance(wafer_map, Image.Image):
        wafer_resized = wafer_map.convert('L').resize(target_dim, Image.NEAREST)
    elif wafer_map.ndim == 2:
        wafer_resized = Image.fromarray(wafer_map.astype(np.uint8)).resize(target_dim, Image.NEAREST)
    else: return np.zeros(required_size)
    X_flat = np.array(wafer_resized).flatten()
    current_size = len(X_flat)
    if current_size < required_size: return np.pad(X_flat,(0,required_size-current_size),'constant')
    return X_flat[:required_size]

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects","Model Insights","About Project"])

with tabs[0]:
    st.header("Choose Model for Prediction")
    model_choice = st.radio("Select model type:", ["CNN (Image-Based)", "XGBoost (Feature-Based)"])

    if model_choice == "CNN (Image-Based)" and cnn_pipe:
        st.subheader("Upload wafer images (.npy or .png/.jpg/.jpeg)")
        uploaded_files = st.file_uploader("Upload wafer maps", type=["png","jpg","jpeg","npy"], accept_multiple_files=True)
        if uploaded_files:
            results = []
            for file in uploaded_files:
                wafer_data_for_display = None
                wafer_for_cnn_predict = None
                try:
                    if file.name.endswith(".npy"):
                        wafer_data_for_display = np.load(file)
                        wafer_for_cnn_predict = Image.fromarray(wafer_data_for_display.astype(np.uint8))
                    else:
                        img = Image.open(file).convert("L")
                        wafer_for_cnn_predict = img
                        wafer_data_for_display = np.array(img)
                except Exception as e:
                    st.error(f"Failed to load {file.name}: {e}")
                    continue
                try:
                    if wafer_for_cnn_predict:
                        label, probs = cnn_pipe.predict(wafer_for_cnn_predict)
                        results.append({
                            "File": file.name,
                            "Predicted_Label": label,
                            "Probabilities": probs,
                            "Wafer_Data": wafer_data_for_display
                        })
                except Exception as e:
                    st.error(f"Prediction failed for {file.name}: {e}")
            st.session_state.cnn_results = results
            st.session_state.cnn_index = 0

        # Display results
        num_results = len(st.session_state.cnn_results)
        if num_results > 0:
            idx = st.slider(f"Select image (1-{num_results})", min_value=1, max_value=num_results, value=1) - 1
            st.session_state.cnn_index = idx
            r = st.session_state.cnn_results[idx]
            wafer_rgb = map_wafer_to_rgb(r['Wafer_Data'])
            st.image(wafer_rgb, width=200, caption=f"File {idx+1} of {num_results}: {r['File']}")
            st.markdown(f"**Predicted:** <span style='color:#ff0000;'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
            st.subheader("Prediction Probabilities")
            for label, prob in sorted(r['Probabilities'].items(), key=lambda x:x[1], reverse=True):
                st.progress(np.clip(prob,0,1))
                st.markdown(f"**{label}**: {prob:.2f}")

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
                    wafer_input = Image.open(file).convert("L")
                X_feat = prepare_pixel_features_for_xgb(wafer_input).reshape(1,-1)
                X_scaled = scaler.transform(X_feat)
                pred_idx = int(xgb.predict(X_scaled)[0])
                pred_label = inv_mapping.get(pred_idx,f"Unknown ({pred_idx})")
                results.append({"File":file.name,"Predicted_Label":pred_label,"Raw_Pred":pred_idx})
            st.session_state.xgb_results = results
            st.subheader("XGBoost Predictions:")
            for r in results:
                st.markdown(f"**{r['File']} → {r['Predicted_Label']}**")

with tabs[1]:
    st.header("Model Insights")
    if st.session_state.cnn_results:
        idx = st.session_state.cnn_index
        r = st.session_state.cnn_results[idx]
        st.markdown(f"### Analysis for Wafer: **{r['File']}**")
        st.markdown(f"**Primary Prediction:** <span style='color:#ff0000;'>{r['Predicted_Label']}</span>", unsafe_allow_html=True)
        if isinstance(r['Probabilities'], dict):
            prob_df = pd.DataFrame({'Defect Type': list(r['Probabilities'].keys()), 'Confidence': r['Probabilities'].values()}).sort_values(by='Confidence', ascending=False)
            st.subheader("Confidence Distribution Across Defect Classes")
            st.bar_chart(prob_df.set_index('Defect Type'), height=350)
    else:
        st.warning("Run a prediction first.")

with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps
    - **XGBoost** for feature-based wafer data
    - **Streamlit** for interactive dashboards
    The **Focal Loss** function was used to train the CNN, improving accuracy on hard-to-classify defects.
    """)
    st.image("https://placehold.co/600x400/1e293b/f8fafc?text=Example+Wafer+Defect+Map", width=300)