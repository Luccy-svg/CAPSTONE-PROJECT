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
