# CAPSTONE-PROJECT

## Dataset

- The dataset `WM811K_fixed.pkl` is too large for GitHub.
You can download it here:
👉 [Download from OneDrive](https://drive.google.com/file/d/1j66GHqpaBWrzcqzLQCJl5J6BrGLmoGRD/view?usp=drive_link)
- The data set was redueced to 300,000 rows from 811,457 rows for easier workflow and less run-time this will not affect the train_test_split.
you can download the new sample here:
👉 [Download from OneDrive](https://drive.google.com/file/d/1pM9vI-hAyDHd7F3qksNJnnpcEMfw8gwp/view?usp=drive_link)

# **WAFER DETECTION** 

# 1. Business Understanding

## 1.1 Background

Semiconductor manufacturing is a highly complex and capital-intensive process involving hundreds of fabrication steps that must be performed with extreme precision. Even microscopic defects introduced during wafer processing can lead to complete product failure, reducing manufacturing yield and increasing production costs. Traditionally, quality control in semiconductor fabrication has relied on manual inspection and rule-based systems, which are time-consuming, subjective, and often unable to keep up with modern production speeds.

A die, in the context of semiconductors, is a small block of the wafer on which a given functional circuit is fabricated. The wafer is cut into many pieces, each containing one copy of the circuit. Each of these pieces is called a die.

In recent years, semiconductor companies such as Intel, TSMC, and Samsung,Nvidia have shifted toward AI-driven defect detection systems to improve yield prediction, defect localization, and root-cause analysis. 

Leveraging machine learning and computer vision, these systems can detect defect patterns directly from wafer map images, enabling earlier and more accurate interventions in the production line.

## 1.2 Problem Statement

Manufacturers need an efficient and automated method to identify and classify wafer defects early in the production process. Manual inspection systems fail to scale with high-volume production and cannot accurately identify subtle, complex defect patterns. Therefore, the goal is to develop a machine learning-based image analysis model capable of automatically detecting and classifying defect patterns in wafer maps therefore improving yield, reducing inspection time, and minimizing production losses.

## 1.3 Business Objective

The primary business objective is to enhance production efficiency and quality assurance in semiconductor manufacturing by automating defect detection. The system will:

Identify wafer defect types using image-based pattern recognition.

Support process engineers in diagnosing the root cause of production faults.

Reduce manual inspection time and related operational costs.

Improve yield rate and product reliability.

Ultimately, the project aims to demonstrate how AI-based defect detection can improve decision-making, reduce downtime, and ensure data-driven manufacturing optimization.

## 1.4 Project Goal

To build and deploy a deep learning-based image classification model capable of identifying common wafer defect patterns (e.g., center, edge-ring, scratch, random) using the WM811K dataset. The model’s predictions will be integrated into an interactive Streamlit dashboard, allowing users to:

Upload wafer map images,

View real-time defect classification and confidence levels,

Visualize feature importance or activation maps (Grad-CAM) for interpretability.

## 1.5 Expected Business Impact

Operational Efficiency: Faster and more accurate defect detection compared to manual methods.

Cost Reduction: Reduced labor costs and fewer defective chips reaching final testing.

Quality Improvement: Early detection minimizes yield loss and improves product reliability.

Decision Support: Data-driven insights for process optimization and predictive maintenance.

Scalability: System can be integrated into production pipelines and scaled to new wafer types.

## 1.6 Success Metrics

Accuracy / F1 Score of classification model

Reduction in defect inspection time by .

Improved detection of rare defect patterns (using confusion matrix or recall metrics).

Usability feedback from engineers or end-users on the Streamlit dashboard prototype.

# 2. Data Understanding

## 2.1 Data Source

The dataset used for this project is the WM811K Wafer Map Dataset, originally released by Taiwan Semiconductor Manufacturing Company (TSMC) and publicly available on sources such as Kaggle and UCI Machine Learning Repository. We downloaded the data from a public dataset, Multimedia Information Retrieval (MIR) lab (http://mirlab.org/dataset/public/).

It consists of wafer map images and corresponding defect labels, representing real-world yield management data collected during semiconductor fabrication processes.

Data-Name: WM811K (Wafer Map Defect Dataset).

Records: 811,457 wafer samples.

Features: Image-based (waferMap) + metadata (dieSize,failureType,lotName,trainTestLabel,waferIndex).

Task Type: Image classification.

Data Format: Pickled or structured array (NumPy, .pkl), and optionally .png images after transformation.

## 2.2 Data Description

Each wafer map image represents a semiconductor wafer subdivided into multiple die. The dataset contains wafers categorized into one of several defect pattern types or labeled as normal (no defect).

Common defect types include:

## 2.3 Data structure and attributes.

Each record contains:

WaferIndex: Unique identifier for each wafer.

LotName: Production batch number, representing wafers produced under the same process conditions.

FailureType: Categorical variable indicating defect type.

Wafer Map Image: 2D array (typically 26×26 or 30×30) where each cell represents a die and its pass/fail status.

Optional Metadata: May include process step, tool ID, or sensor readings depending on version.

## 2.4 Data Quality and Challenges

### Aspect Observation

**Missing Values**: Some wafers have incomplete data (no label or image mismatch).

**Imbalanced Classes**: Majority of wafers are labeled as “None” (normal), making minority defect patterns rare.

**Noise**: Random noise due to manufacturing variability and sensor differences.

Data Format: Some datasets are stored as .pkl (Pickle) files that require unpickling and image reconstruction.

### Mitigation Strategies:

Handle missing or corrupted wafer maps by filtering invalid entries.

Apply data augmentation to balance rare defect types.

Use normalization and reshaping for consistent image input size.

Visualize samples per class before training to guide resampling.

## 2.5 Initial Data Exploration Goals

Visualize a few wafer maps for each defect type to understand pattern structure.

Check Class Distribution: Verify imbalance between defect categories.

Inspect Image Dimensions: Ensure uniformity across all wafers.

Validate Data Integrity: Confirm each wafer has both an image and a label.

Compute Basic Statistics: Such as pixel intensity distributions and label frequencies.

## 2.6 Summary

The dataset provides a rich, realistic simulation of semiconductor manufacturing defects.

It combines computer vision challenges (pattern recognition, noise handling) with predictive modeling needs (classification, imbalance learning).

Understanding the data structure and patterns will be key to designing robust models for defect detection and yield prediction, aligning both with technical depth and business objectives.

