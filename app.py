import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from cnn_pipeline import WaferCNNPipeline

# -------------------- FILE PATHS -------------------- #
CNN_MODEL_PATH = "cnn_model.keras"
XGB_PATH = "xgboost_improved.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DEMO_PATH = "demo_data"
DEMO_IMAGES = os.path.join(DEMO_PATH, "images")
DEMO_YTEST = os.path.join(DEMO_PATH, "y_test.npy")
DEMO_LABEL_ENCODER = os.path.join(DEMO_PATH, "label_encoder.pkl")
DEMO_XGB = os.path.join(DEMO_PATH, "xgb_demo.csv")

# -------------------- GENERATE DEMO DATA -------------------- #
def generate_demo_data():
    os.makedirs(DEMO_IMAGES, exist_ok=True)
    defect_types = ["Edge-Ring", "Center", "Loc", "Scratch", "Random"]
    failure_num_enc = {ftype: i for i, ftype in enumerate(defect_types)}

    # CNN demo images
    for i, ftype in enumerate(defect_types):
        wafer = np.zeros((32,32), dtype=np.float32)
        if ftype == "Center": wafer[12:20,12:20] = 1.0
        elif ftype == "Edge-Ring": wafer[0,:] = wafer[-1,:] = wafer[:,0] = wafer[:,-1] = 1.0
        elif ftype == "Loc": wafer[8:12,8:12] = 1.0
        elif ftype == "Scratch": wafer[16:18,:] = 1.0
        elif ftype == "Random": wafer[np.random.randint(0,32,5), np.random.randint(0,32,5)] = 1.0
        Image.fromarray((wafer*255).astype(np.uint8)).save(f"{DEMO_IMAGES}/wafer_{i}_{ftype}.png")
        np.save(f"{DEMO_IMAGES}/wafer_{i}_{ftype}.npy", wafer)

    # CNN labels
    df_cnn = pd.DataFrame({
        "file": [f"wafer_{i}_{ftype}.npy" for i, ftype in enumerate(defect_types)],
        "failureType": defect_types,
        "failureNum_enc": [failure_num_enc[f] for f in defect_types]
    })
    df_cnn.to_csv(os.path.join(DEMO_PATH,"cnn_labels.csv"), index=False)

    # XGBoost demo features (1029 features)
    num_features = 1029
    X_demo = np.random.rand(len(defect_types), num_features)
    y_demo = np.array([failure_num_enc[f] for f in defect_types])
    df_xgb = pd.DataFrame(X_demo, columns=[f"feature_{i}" for i in range(num_features)])
    df_xgb["true_label"] = y_demo
    df_xgb.to_csv(DEMO_XGB, index=False)

    # Labels for insights
    np.save(DEMO_YTEST, y_demo)
    np.save(os.path.join(DEMO_PATH,"y_test_pred.npy"), y_demo)

    # Label Encoder
    le = LabelEncoder()
    le.fit(defect_types)
    joblib.dump(le, DEMO_LABEL_ENCODER)

# Auto-generate demo if missing
if not os.path.exists(DEMO_XGB) or not os.path.exists(DEMO_YTEST):
    generate_demo_data()

# -------------------- STREAMLIT CONFIG -------------------- #
st.set_page_config(page_title="ChipSleuth: Wafer Defect Dashboard", page_icon="🕵️‍♀️", layout="wide")
st.title("ChipSleuth – Semiconductor Wafer Defect Detection")

# -------------------- SESSION STATE -------------------- #
for key in ["cnn_results","xgb_results","wafer_index","xgb_index"]:
    if key not in st.session_state: st.session_state[key] = None if "results" in key else 0

# -------------------- LOAD MODELS -------------------- #
cnn_pipe = WaferCNNPipeline(CNN_MODEL_PATH, LABEL_ENCODER_PATH) if os.path.exists(CNN_MODEL_PATH) else None
xgb = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# -------------------- UTILITY -------------------- #
def map_label(label):
    return "No Defect" if label == "0 0" else str(label)

def show_wafer(file, label, probs):
    wafer = np.load(file)
    wafer_img = Image.fromarray((wafer*255).astype(np.uint8)).resize((128,128), Image.NEAREST)
    st.image(wafer_img, caption=f"{os.path.basename(file)} → {label}")
    st.bar_chart(pd.DataFrame([probs]))

# -------------------- TABS -------------------- #
tabs = st.tabs(["Predict Defects", "Model Insights", "About Project"])

# -------------------- TAB 1: PREDICTION -------------------- #
with tabs[0]:
    st.header("Choose Model Type for Prediction")
    model_choice = st.radio("Select model type:", ["XGBoost (Feature-Based)", "CNN (Image-Based)"])

    # ---------- Demo Data ----------
    st.subheader("Or use Demo Data")
    if st.button("Load Demo Data"):
        if model_choice == "XGBoost (Feature-Based)":
            df = pd.read_csv(DEMO_XGB)
            X_scaled = scaler.transform(df.drop(columns=["true_label"]).values)
            preds = xgb.predict(X_scaled)
            df["Predicted Defect"] = [map_label(p) for p in preds]
            st.session_state.xgb_results = df
            st.dataframe(df, use_container_width=True)
            st.download_button("Download XGBoost Predictions", df.to_csv(index=False).encode("utf-8"), "xgb_predictions.csv")

        elif model_choice == "CNN (Image-Based)":
            demo_files = sorted([os.path.join(DEMO_IMAGES, f) for f in os.listdir(DEMO_IMAGES) if f.endswith(".npy")])
            results = []
            for file in demo_files:
                label, probs = cnn_pipe.predict(np.load(file))
                results.append({"File": os.path.basename(file), "Predicted_Label": map_label(label), "Probabilities": probs})
            st.session_state.cnn_results = results
            st.session_state.current_idx = 0

            # Navigation buttons
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                if st.button("Previous"):
                    st.session_state.current_idx = max(0, st.session_state.current_idx-1)
            with col3:
                if st.button("Next"):
                    st.session_state.current_idx = min(len(demo_files)-1, st.session_state.current_idx+1)

            # Show wafer
            idx = st.session_state.current_idx
            show_wafer(demo_files[idx], results[idx]["Predicted_Label"], results[idx]["Probabilities"])
            st.download_button("Download CNN Predictions", pd.DataFrame([{"File": r["File"], "Predicted_Label": r["Predicted_Label"]} for r in results]).to_csv(index=False).encode("utf-8"), "cnn_predictions.csv")

    # ---------- Upload real images ----------
    uploaded_files = st.file_uploader("Upload wafer maps (.png, .jpg, .jpeg, .npy)", type=["png","jpg","jpeg","npy"], accept_multiple_files=True)
    if uploaded_files and cnn_pipe:
        results = []
        for f in uploaded_files:
            wafer = np.load(f) if f.name.endswith(".npy") else np.array(Image.open(f).convert("L"))
            label, probs = cnn_pipe.predict(wafer)
            results.append({"File": f.name, "Predicted_Label": map_label(label), "Probabilities": probs})
        st.session_state.cnn_results = results
        for r, f in zip(results, uploaded_files):
            wafer = np.load(f) if f.name.endswith(".npy") else np.array(Image.open(f).convert("L"))
            wafer_img = Image.fromarray((wafer*255).astype(np.uint8)).resize((128,128), Image.NEAREST)
            st.image(wafer_img, caption=f"{f.name} → {r['Predicted_Label']}")
            st.bar_chart(pd.DataFrame([r["Probabilities"]]))
        st.download_button("Download CNN Predictions", pd.DataFrame([{"File": r["File"], "Predicted_Label": r["Predicted_Label"]} for r in results]).to_csv(index=False).encode("utf-8"), "cnn_predictions.csv")

# -------------------- TAB 2: MODEL INSIGHTS -------------------- #
with tabs[1]:
    st.header("Model Insights")
    model_choice = st.selectbox("Select Model to View Insights", ["XGBoost", "CNN"])

    # ---------- XGBoost Insights ----------
    if model_choice == "XGBoost" and st.session_state.xgb_results is not None:
        df = st.session_state.xgb_results
        is_demo = df["true_label"].max() < len(le.classes_)
        if is_demo:
            y_true = [map_label(str(y)) for y in df["true_label"].values]
            y_pred = [map_label(str(d)) for d in df["Predicted Defect"].values]
        else:
            y_true = df["true_label"].values
            y_pred = le.transform(df["Predicted Defect"].values)
        classes_in_data = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=classes_in_data)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes_in_data, yticklabels=classes_in_data)
        st.pyplot(plt)
        plt.close()
        report = classification_report(y_true, y_pred, labels=classes_in_data, target_names=classes_in_data, output_dict=True)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    # ---------- CNN Insights ----------
    elif model_choice == "CNN" and st.session_state.cnn_results:
        y_true_raw = np.load(DEMO_YTEST)
        y_true = [map_label(str(y)) for y in y_true_raw]
        y_pred = [r["Predicted_Label"] for r in st.session_state.cnn_results]
        classes_in_data = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=classes_in_data)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes_in_data, yticklabels=classes_in_data)
        st.pyplot(plt)
        plt.close()
        report = classification_report(y_true, y_pred, labels=classes_in_data, target_names=classes_in_data, output_dict=True)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.info("Run or load predictions first to view insights.")

# -------------------- TAB 3: ABOUT -------------------- #
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This project detects **semiconductor wafer defects** using:
    - **CNN** for image-based wafer maps  
    - **XGBoost** for feature-based wafer data  
    - **Streamlit** for an interactive dashboard deployment  

    **Goal:** Automate defect detection and enhance wafer yield prediction.
    """)