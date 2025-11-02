# Suppress warnings and logs for clean Streamlit output
import warnings
warnings.filterwarnings("ignore")  
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

# Core libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import gdown
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from cnn_pipeline import WaferCNNPipeline


# CONFIGURATION

st.set_page_config(page_title="Wafer Defect Classifier", layout="wide")
st.title("Semiconductor Wafer Defect Detection Dashboard")
st.sidebar.header("Model Selection")
st.toast("Models loaded successfully! Ready to classify wafers")

tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])


# LOAD MODELS (RF, XGB, CNN, etc.)


# Paths and model setup
MODEL_PATH = "random_forest_improved.pkl"
DRIVE_FILE_ID = "1T6Ox-zPpgW5npnN7Cl1m7uskLUyZCEw5"

# Safe custom loader for Random Forest
def load_rf_model():
    """Download and load Random Forest model if not found locally."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Random Forest model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    model = joblib.load(MODEL_PATH)
    return model



# CNN model load
CNN_MODEL_PATH = "cnn_model.h5"
if not os.path.exists(CNN_MODEL_PATH):
    gdrive_url = "https://drive.google.com/uc?id=1cUp_ZBRcz2Eu6Q76X3HeogGzrTW_4KxM"
    gdown.download(gdrive_url, CNN_MODEL_PATH, quiet=False)

cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, "label_encoder.pkl")

# Load models and utilities
rf = load_rf_model()
xgb = joblib.load("xgboost_improved.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
cnn_pipe = WaferCNNPipeline("cnn_model.h5", "label_encoder.pkl")


# TAB 1: PREDICTION

with tabs[0]:
    st.header("Choose Model Type for Prediction")

    model_choice = st.radio(
        "Select model type:",
        ["Improved Random Forest (Feature-Based)", "CNN (Image-Based)"]
    )

    
    # FEATURE-BASED MODELS (RF / XGB)
    
    if model_choice == "Improved Random Forest (Feature-Based)":
        st.subheader("Upload CSV for Feature-Based Prediction")

        selected_model = st.radio("Select model:", ["Improved Random Forest", "XGBoost"])
        uploaded_csv = st.file_uploader("Upload wafer features (.csv)", type=["csv"])

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), use_container_width=True)

            X_scaled = scaler.transform(df.values)

            # Choose model
            model = rf if selected_model == "Improved Random Forest" else xgb
            preds = model.predict(X_scaled)
            decoded = le.inverse_transform(preds)
            df["Predicted Defect"] = decoded

            st.success(f"Prediction complete using {selected_model}!")
            st.dataframe(df[["Predicted Defect"]])

    # -----------------------------------------------------------------
    # IMAGE-BASED MODEL (CNN)
    # -----------------------------------------------------------------
    else:
        st.subheader("Upload Wafer Map Images for CNN Prediction")
        uploaded_files = st.file_uploader(
            "Upload wafer maps (.png, .jpg, .jpeg, .npy)",
            type=["png", "jpg", "jpeg", "npy"],
            accept_multiple_files=True
        )

        if uploaded_files:
            results = []

            for uploaded_file in uploaded_files:
                # Preprocess
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    img = Image.open(uploaded_file).convert("L").resize((26, 26))
                    wafer = np.array(img) / 255.0

                wafer_input = wafer.reshape(1, 26, 26, 1)

                preds = cnn_pipe.model.predict(wafer_input, verbose=0)
                pred_class = np.argmax(preds, axis=1)
                label = cnn_pipe.le.inverse_transform(pred_class)[0]
                probs = preds[0]

                results.append({
                    "File": uploaded_file.name,
                    "Predicted_Label": label,
                    "Probabilities": dict(zip(cnn_pipe.le.classes_, probs))
                })

            st.subheader("Prediction Results")
            df_results = pd.DataFrame([{
                "File": r["File"],
                "Predicted_Label": r["Predicted_Label"]
            } for r in results])
            st.dataframe(df_results, use_container_width=True)

            # Show images + probability bars
            for r, uploaded_file in zip(results, uploaded_files):
                wafer = (
                    np.load(uploaded_file)
                    if uploaded_file.name.endswith(".npy")
                    else np.array(Image.open(uploaded_file).convert("L").resize((26, 26))) / 255.0
                )

                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].imshow(wafer, cmap="gray")
                ax[0].set_title(f"{uploaded_file.name}\nPredicted: {r['Predicted_Label']}")
                ax[0].axis("off")

                ax[1].barh(list(r["Probabilities"].keys()), list(r["Probabilities"].values()))
                ax[1].set_title("Class Probabilities")
                st.pyplot(fig)
                plt.close(fig)

            st.balloons()
        else:
            st.info("Upload wafer map images to start predictions.")


# TAB 2: MODEL INSIGHTS

with tabs[1]:
    st.header("Model Insights")
    model_choice = st.selectbox("Select Model to View Insights", ["Random Forest", "XGBoost", "CNN"])

    if model_choice in ["Random Forest", "XGBoost"]:
        model_file = "random_forest_improved.pkl" if model_choice == "Random Forest" else "xgboost_improved.pkl"
        model = joblib.load(model_file)

        try:
            y_test_pred = joblib.load(f"{model_choice.lower()}_y_test_pred.pkl")
            y_test_true = joblib.load("y_test.pkl")
        except FileNotFoundError:
            st.warning("Missing test data files. Add y_test.pkl and *_y_test_pred.pkl for insights.")
            st.stop()

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_true, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        st.pyplot(plt)

        st.subheader("Classification Report")
        report = classification_report(y_test_true, y_test_pred, target_names=le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        if model_choice == "Random Forest":
            st.subheader("Top 10 Feature Importances")
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]
            plt.figure(figsize=(8, 5))
            plt.barh(range(len(indices)), importances[indices], color="skyblue")
            plt.yticks(range(len(indices)), np.array([f"Feature {i}" for i in indices]))
            plt.xlabel("Importance")
            plt.title("Top Features")
            st.pyplot(plt)

    elif model_choice == "CNN":
        st.write("🧩 CNN learns directly from wafer image patterns rather than numerical features.")
        st.image("cnn_filters_example.png", caption="Example learned CNN filters", use_container_width=True)


# TAB 3: ABOUT PROJECT

with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps  
    - **Random Forest** and **XGBoost** for feature-based wafer data  
    - **SMOTE** for synthetic balancing of minority defect classes  
    - **Streamlit** for an interactive dashboard deployment  

    **Goal:** Automate defect detection and enhance wafer yield prediction.
    """)