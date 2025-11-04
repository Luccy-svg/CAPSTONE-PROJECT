
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model # Using keras from tensorflow is more robust

# FOCAL LOSS DEFINITION
# This is required to correctly load a model that was compiled with a custom loss.
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function used during training."""
    def focal_loss_fixed(y_true, y_pred):
        # Clip y_pred to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8) 
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Weighting factor: alpha * (1 - p)^gamma
        weight = alpha * tf.math.pow(1 - y_pred, gamma) 
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: handles model loading, preprocessing, 
    and prediction for wafer map images.
    """

    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32, 32)):
        # CRITICAL FIX 2: Pass custom_objects to load_model 
        # Defines focal_loss upon loading to correctly deserialize the model structure.
        self.model = load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss()}, 
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image: Image.Image) -> np.ndarray:
        """
        Accepts PIL Image, returns processed image tensor.
        Robustly converts PIL Image to a 32x32 array with values 0.0, 1.0, and 2.0.
        """
        # 1. Validation and Initial Conversion
        if not isinstance(wafer_image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        # Convert to grayscale and resize using nearest neighbor for pattern retention
        wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
        wafer_array = np.array(wafer_image, dtype=np.float32)

        # 2. Rescaling Wafer Map Data
        # We assume 0/1/2 original values, but image conversion (PNG/JPEG) often introduces 0-255 scaling.
        
        max_val = wafer_array.max()

        # Robust two-step thresholding for non-zero values if input is scaled (0-255)
        if max_val > 2.0:
            unique_non_zero = np.unique(wafer_array[wafer_array > 0])
            
            if len(unique_non_zero) >= 2:
                # Find the two highest unique non-zero values to set the split point
                sorted_unique = np.sort(unique_non_zero)
                split_point = (sorted_unique[-1] + sorted_unique[-2]) / 2.0
                
                # Re-map: values above the split point become 2.0 (Defect)
                wafer_array[wafer_array > split_point] = 2.0
                # Re-map: values between 0 and split_point become 1.0 (Functional)
                wafer_array[(wafer_array > 0.0) & (wafer_array <= split_point)] = 1.0

            elif len(unique_non_zero) == 1:
                # Only one non-zero value present. Map all non-zero to 1.0.
                wafer_array[wafer_array > 0.0] = 1.0
            
            # Values of 0.0 are already correct (background)
        
        # 3. Final Reshape
        wafer_array = wafer_array.astype(np.float32)

        # Reshape for CNN (Add batch and channel dimensions: [1, 32, 32, 1])
        wafer_array = wafer_array.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_array

    def predict(self, wafer_image: Image.Image):
        """
        Returns predicted label and probabilities dictionary for known classes.
        """
        x = self.preprocess(wafer_image)
        # Use verbose=0 for cleaner output
        preds = self.model.predict(x, verbose=0)[0] 

        pred_class = int(np.argmax(preds))

        # Safely handle class index mapping
        if pred_class < len(self.le.classes_):
            label = self.le.inverse_transform([pred_class])[0]
        else:
            # Fallback for robustness
            label = self.le.classes_[np.argmax(preds[:len(self.le.classes_)])] 

        # Probabilities dictionary
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) for i in range(len(self.le.classes_))}
        return label, probs
