import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os

# Create folders for demo
os.makedirs("demo_data", exist_ok=True)
os.makedirs("demo_data/images", exist_ok=True)

# Define defect types
defect_types = ["Edge-Ring", "Center", "Loc", "Scratch", "Random"]
failure_num_enc = {ftype: i for i, ftype in enumerate(defect_types)}

# Generate synthetic wafer images
wafer_images = []
labels = []
for i, ftype in enumerate(defect_types):
    wafer = np.zeros((32,32), dtype=np.float32)
    
    if ftype == "Center":
        wafer[12:20, 12:20] = 1.0
    elif ftype == "Edge-Ring":
        wafer[0,:] = wafer[-1,:] = wafer[:,0] = wafer[:,-1] = 1.0
    elif ftype == "Loc":
        wafer[8:12, 8:12] = 1.0
    elif ftype == "Scratch":
        wafer[16:18, :] = 1.0
    elif ftype == "Random":
        wafer[np.random.randint(0,32,5), np.random.randint(0,32,5)] = 1.0
    
    wafer_images.append(wafer)
    labels.append(ftype)
    
    # Save image as PNG
    img = Image.fromarray((wafer*255).astype(np.uint8))
    img.save(f"demo_data/images/wafer_{i}_{ftype}.png")
    
    # Also save as .npy for CNN input
    np.save(f"demo_data/images/wafer_{i}_{ftype}.npy", wafer)

# Save CNN labels
df_cnn = pd.DataFrame({"file": [f"wafer_{i}_{ftype}.npy" for i, ftype in enumerate(defect_types)],
                       "failureType": labels,
                       "failureNum_enc": [failure_num_enc[f] for f in labels]})
df_cnn.to_csv("demo_data/cnn_labels.csv", index=False)

# Generate mock XGBoost features (e.g., 10 features per wafer)
X_demo = np.random.rand(len(defect_types), 10)
y_demo = np.array([failure_num_enc[f] for f in labels])

# Save XGBoost test CSV
df_xgb = pd.DataFrame(X_demo, columns=[f"feature_{i}" for i in range(10)])
df_xgb["true_label"] = y_demo
df_xgb.to_csv("demo_data/xgb_features.csv", index=False)

# Save labels for insights
np.save("demo_data/y_test.npy", y_demo)
np.save("demo_data/y_test_pred.npy", y_demo)  # initially set predicted = true for demo

# Optional: save label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)
joblib.dump(le, "demo_data/label_encoder.pkl")

print("Demo dataset created in 'demo_data/' folder.")

