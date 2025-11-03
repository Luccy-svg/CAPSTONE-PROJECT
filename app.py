# Suppress warnings and logs
import warnings
warnings.filterwarnings("ignore")  
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Core libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from cnn_pipeline import WaferCNNPipeline

# -------------------- CONFIGURATION -------------------- #
st.set_page_config(page_title="Wafer Defect Classifier", layout="wide")
st.title("Semiconductor Wafer Defect Detection Dashboard")
st.sidebar.header("Model Selection")
st.toast("Models loaded successfully! Ready to classify wafers")

# -------------------- FILE CHECKS -------------------- #
st.write("Files in working directory:", os.listdir())
st.write("CNN exists:", os.path.exists("cnn_model.keras"))

tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- LOAD MODELS -------------------- #

# XGBoost + Utilities
xgb = joblib.load("xgboost_improved.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# CNN model (from Git repo)
CNN_MODEL_PATH = "cnn_model.keras"  # Already in repo
if not os.path.exists(CNN_MODEL_PATH):
    st.error(f"Model file not found: {CNN_MODEL_PATH}. Please check your repository.")
else:
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, "label_encoder.pkl")

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model Type for Prediction")

    model_choice = st.radio(
        "Select model type:",
        ["XGBoost (Feature-Based)", "CNN (Image-Based)"]
    )

    # ---------------- FEATURE-BASED (XGBoost) ---------------- #
    if model_choice == "XGBoost (Feature-Based)":
        st.subheader("Upload CSV for Feature-Based Prediction")
        uploaded_csv = st.file_uploader("Upload wafer features (.csv)", type=["csv"])

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), use_container_width=True)

            X_scaled = scaler.transform(df.values)
            preds = xgb.predict(X_scaled)
            decoded = le.inverse_transform(preds)
            df["Predicted Defect"] = decoded

            st.success("Prediction complete using XGBoost!")
            st.dataframe(df[["Predicted Defect"]])

    # ---------------- IMAGE-BASED (CNN) ---------------- #
    else:
        st.subheader("Upload Wafer Map Images for CNN Prediction")
        uploaded_files = st.file_uploader(
            "Upload wafer maps (.png, .jpg, .jpeg, .npy)",
            type=["png", "jpg", "jpeg", "npy"],
            accept_multiple_files=True
        )

        if uploaded_files and cnn_pipe is not None:
            results = []

            for uploaded_file in uploaded_files:
                # ---------------- Preprocessing ---------------- #
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    img = Image.open(uploaded_file).convert("L").resize((32, 32))
                    wafer = np.array(img) / 255.0  # normalize to [0,1]

                wafer_input = wafer.reshape(1, 32, 32, 1)

                # ---------------- Prediction ---------------- #
                preds = cnn_pipe.model.predict(wafer_input, verbose=0)
                pred_class = np.argmax(preds, axis=1)
                label = cnn_pipe.le.inverse_transform(pred_class)[0]
                probs = preds[0]

                results.append({
                    "File": uploaded_file.name,
                    "Predicted_Label": label,
                    "Probabilities": dict(zip(cnn_pipe.le.classes_, probs))
                })

            # ---------------- Display Results ---------------- #
            st.subheader("Prediction Results")
            df_results = pd.DataFrame([{
                "File": r["File"],
                "Predicted_Label": r["Predicted_Label"]
            } for r in results])
            st.dataframe(df_results, use_container_width=True)

            # Show images + probability bars
            for r, uploaded_file in zip(results, uploaded_files):
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    img = Image.open(uploaded_file).convert("L").resize((32, 32))
                    wafer = np.array(img) / 255.0

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

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights")
    model_choice = st.selectbox("Select Model to View Insights", ["XGBoost", "CNN"])

    if model_choice == "XGBoost":
        try:
            y_test_pred = joblib.load("xgboost_y_test_pred.pkl")
            y_test_true = joblib.load("y_test.pkl")
        except FileNotFoundError:
            st.warning("Missing test data files. Add y_test.pkl and xgboost_y_test_pred.pkl for insights.")
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

    elif model_choice == "CNN":
        st.write("🧩 CNN learns directly from wafer image patterns rather than numerical features.")
        st.image("cnn_filters_example.png", caption="Example learned CNN filters", use_container_width=True)

# -------------------- TAB 3: ABOUT PROJECT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps  
    - **XGBoost** for feature-based wafer data  
    - **SMOTE** for synthetic balancing of minority defect classes  
    - **Streamlit** for an interactive dashboard deployment  

    **Goal:** Automate defect detection and enhance wafer yield prediction.
    """)