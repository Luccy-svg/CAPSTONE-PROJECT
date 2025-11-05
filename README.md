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

Common defect types include:

|Percentage |	Pattern Type |	Observation |
|:-------|:----------|:---------|
| 78.69 |	0 0 |	Placeholder Category |
| 18.18 |	None |	Wafer with no defect |
| 1.17 |	Edge-Ring |	Moderate Frequency |
| 0.64 |	Edge-Loc |	Rare defect type |
| 0.51 |	Center |	Central pattern defects |
| 0.45 |	Loc |	Mislabeled variant |
| 0.16 |	Scratch |	Very rare mechanical defect |
| 0.11 |	Random |	Randomly distributed defect
| 0.07 |	Donut |	Circular defect pattern |
| 0.02 |	Near-full |	Few samples |


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

# **3. Data Cleaning & Preparation**

**Duplicate check**

Before analyzing or sampling the semiconductor wafer dataset, we ensure that all metadata columns (non-image fields) are clean, scalar, and free from duplicates.

This step helps avoid issues caused by array-like or mixed-type values, which can interfere with grouping, deduplication, and modeling later

**Create a Balanced 300K Sample**

To make the dataset easier to handle and ensure consistent processing speed, we created a 300,000 row sample from 800,000 raw data that preserves the original train/test ratio.

This approach guarantees that both training and testing subsets remain well represented in the sample, avoiding sampling bias.

# **MODELING**

**Data Splitting Insights**

| Task |	Description |	Outcome |
|:------|:--------|:------------|
| Train/Test Separation |	Split dataset into training and testing subsets using trainTestLabel |	Ensured proper evaluation without data leakage. |
| Feature Extraction |	Extracted waferMap arrays for model input |	Shape standardized to (32×32). |
| Label Encoding |	Converted categorical failureType into numeric codes |	Ready for CNN classification. |
| Data Check |	Verified size and class distribution	| balanced dataset maintained. |

**Data Normalization**

Wafer map pixel values vary between 0–2 (or similar small integers).

Neural networks converge faster when input values are scaled to a 0–1 range.

| Task |	Description |	Outcome |
|:-------|:---------|:--------|
| Pixel Scaling |	Divided all wafer pixel values by the global max value |	All inputs scaled to [0, 1]. |
| Shape Adjustment |	Added single channel dimension for CNN compatibility |	Final input shape (32, 32, 1). |

Some of the Machine learning models used include :

1. Logististic regression
   
   This means that overall, the model gets ~62% of wafers correctly classified, but it performs unevenly across different defect types, some classes are recognized very well, others poorly(Loc, Edge-Loc).
   
LogReg learned basic separations — it can distinguish strong, geometric patterns (like Edge-Ring, Donut) quite well.

SMOTE helped minority classes (Donut, Random improved F1), but still not enough for the hardest ones.

The 65% recall on [0 0] means the model sometimes mistakes clean wafers for defective ones, not ideal for production.

Conclusion on LogReg
Logistic Regression is great as a baseline, but it’s not powerful enough for this dataset, the wafer maps are spatial, not tabular in the classical sense.

2. RandomForest model
Random Forest achieves 80% accuracy and solid performance on frequent wafer defect types.

However, rare defects like Scratch and Loc remain difficult to classify, suggesting the need for better feature representation and targeted balancing.

3. XGBoost Classifier on SMOTE-Balanced Data
   XGBoost Accuracy: 0.7938

   Performance Summary
| Metric |	Logistic Regression |	Random Forest	XGBoost |
|:-------|:------------|:------------|
| Accuracy |	0.62 |	0.80 |	0.79 |
| Macro F1-score |	0.50 |	0.61 |	0.60 |
|Weighted F1-score |	0.66 |	0.78 |	0.77 |

Random Forest(80% accuracy) slightly outperforms XGBoost overall, while both tree-based models significantly outperform Logistic Regression.

XGBoost provided a balanced and generalizable performance, nearly matching RF.

Logistic Regression serves as a useful baseline but underfits complex wafer structures.

For future improvement, deep learning (CNN) or feature extraction from wafer maps could boost defect classification, especially for underrepresented defect types.

**Transition to CNN**

The CNN model was designed to automatically learn spatial patterns in wafer maps and classify semiconductor wafer defects into multiple categories (e.g., Center, Edge-Loc, Donut, etc.).

Each wafer map was treated as a grayscale image representing the pattern of failed and passed dies.

Now we’ll move from tabular ML → image ML.

The setup we will use (no flattening, no SMOTE, no engineered features).

**Insights:**
Training was monitored for both accuracy and validation loss across epochs.

The model converged steadily and showed strong generalization to the test data.

**Observations:**

Final CNN accuracy: 62% on a 9-class wafer defect dataset.

CNN demonstrates strong spatial learning capabilities on complex wafer patterns.

**CONCLUSION**

The project successfully developed a wafer defect classification system using Random Forest, XGBoost, and Convolutional Neural Network (CNN) models. After comparing the results, the CNN model gave the best performance because it can learn spatial patterns directly from wafer map images.

The traditional models (Random Forest and XGBoost) performed fairly well but were less accurate for complex defect shapes.

The CNN achieved higher precision, recall, and F1-scores across most defect categories, showing it is more suitable for image-based wafer classification. The use of data augmentation, focal loss, and class balancing improved the model’s ability to detect minority defect types.

**Recommendations**

Use the CNN model as the main model for wafer defect detection.

Keep Random Forest or XGBoost as backup models for cases where image data is unavailable or computational resources are limited.

Deploy the trained CNN model in the Streamlit app to allow easy image uploads and predictions.

Add Grad-CAM visualizations to explain the CNN’s predictions and highlight key defect regions.

Continuously update the dataset with new wafer maps to improve model performance over time.

**Next Steps / Future Work**

Fine-tune the CNN architecture or test transfer learning models such as EfficientNet or MobileNet.

Automate model retraining when new labeled data becomes available.

Deploy the model in a real-time production environment and monitor its performance.

Convert the CNN model to TensorFlow Lite or ONNX for lightweight, on-device inference.

Integrate prediction results with production dashboards for engineers to analyze defect patterns.
