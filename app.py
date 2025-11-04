# -------------------- SUPPRESS WARNINGS -------------------- #
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import logging

# Replace tf.get_logger() with Python logger
tf.get_logger = lambda: logging.getLogger('tensorflow')
tf.get_logger().setLevel(logging.ERROR)

# -------------------- IMPORT LIBRARIES -------------------- #
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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
    st.session_state.cnn_results = None

# -------------------- FILE CHECKS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"

st.write("Files in working directory:", os.listdir())
st.write("CNN exists:", os.path.exists(CNN_MODEL_PATH))

# -------------------- LOAD CNN MODEL -------------------- #
if os.path.exists(CNN_MODEL_PATH):
    st.success("CNN model is present.")
    cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH)
else:
    st.error(f"Model file not found: {CNN_MODEL_PATH}")
    cnn_pipe = None

# -------------------- LOAD XGBOOST AND UTILITIES -------------------- #
xgb = joblib.load("xgboost_improved.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------- UTILITY -------------------- #
def map_label(label):
    return "No Defect" if label == "0 0" else label

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model Type for Prediction")
    model_choice = st.radio(
        "Select model type:",
        ["XGBoost (Feature-Based)", "CNN (Image-Based)"]
    )

    # ---------- XGBoost Prediction ---------- #
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
            df["Predicted Defect"] = df["Predicted Defect"].apply(map_label)

            st.success("Prediction complete using XGBoost!")
            st.dataframe(df[["Predicted Defect"]], use_container_width=True)

    # ---------- CNN Prediction ---------- #
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
                # Load wafer image
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    wafer = Image.open(uploaded_file).convert("L")

                label, probs = cnn_pipe.predict(wafer)
                results.append({
                    "File": uploaded_file.name,
                    "Predicted_Label": map_label(label),
                    "Probabilities": probs
                })

            # Save results to session state for insights
            st.session_state.cnn_results = results

            # Display results table
            df_results = pd.DataFrame([{"File": r["File"], "Predicted_Label": r["Predicted_Label"]} for r in results])
            st.dataframe(df_results, use_container_width=True)

            # Show images + probability bars
            for r, uploaded_file in zip(results, uploaded_files):
                if uploaded_file.name.endswith(".npy"):
                    wafer = np.load(uploaded_file)
                else:
                    wafer = Image.open(uploaded_file).convert("L")

                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].imshow(wafer, cmap="gray")
                ax[0].set_title(f"{uploaded_file.name}\nPredicted: {r['Predicted_Label']}")
                ax[0].axis("off")

                ax[1].barh(list(r["Probabilities"].keys()), list(r["Probabilities"].values()))
                ax[1].set_title("Class Probabilities")
                st.pyplot(fig)
                plt.close(fig)

            st.success("CNN Predictions complete!")
        else:
            st.info("Upload wafer map images to start predictions.")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights")
    model_choice = st.selectbox("Select Model to View Insights", ["XGBoost", "CNN"])

    # ---------- XGBoost Insights ---------- #
    if model_choice == "XGBoost":
        st.subheader("XGBoost Insights")
        uploaded_csv = st.file_uploader(
            "Upload CSV of test features (.csv) for XGBoost insights",
            type=["csv"]
        )

        if uploaded_csv is not None:
            df_test = pd.read_csv(uploaded_csv)
            st.write("Test features preview:", df_test.head())

            X_test_scaled = scaler.transform(df_test.values)
            y_test_pred = xgb.predict(X_test_scaled)
            decoded_pred = le.inverse_transform(y_test_pred)

            st.subheader("Predictions Overview")
            st.dataframe(pd.DataFrame({"Predicted": [map_label(d) for d in decoded_pred]}), use_container_width=True)

            uploaded_labels = st.file_uploader("Upload true labels CSV", type=["csv"], key="ytest")
            if uploaded_labels:
                y_true = pd.read_csv(uploaded_labels).values.ravel()
                y_true = [map_label(y) for y in y_true]

                cm = confusion_matrix(y_true, [map_label(d) for d in decoded_pred])
                plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
                st.pyplot(plt)
                plt.close()

                report = classification_report(y_true, [map_label(d) for d in decoded_pred], output_dict=True)
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

    # ---------- CNN Insights ---------- #
    elif model_choice == "CNN":
        st.subheader("CNN Insights")
        st.write("CNN learns directly from wafer image patterns.")

        if st.session_state.cnn_results:
            try:
                y_true = joblib.load("y_test.pkl")
                y_true = [map_label(y) for y in y_true]
                y_pred = [r["Predicted_Label"] for r in st.session_state.cnn_results]

                if len(y_true) != len(y_pred):
                    st.warning("Mismatch in length. Showing available insights.")
                    min_len = min(len(y_true), len(y_pred))
                    y_true = y_true[:min_len]
                    y_pred = y_pred[:min_len]

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
                plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
                st.pyplot(plt)
                plt.close()

                # Classification Report
                report = classification_report(y_true, y_pred, output_dict=True)
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

            except FileNotFoundError:
                st.warning("y_test.pkl not found. Run XGBoost predictions or upload true labels.")

        else:
            st.info("Run CNN predictions first to view insights.")

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