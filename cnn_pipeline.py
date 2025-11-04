
import tensorflow as tf
import numpy as np
import joblib
import io
from PIL import Image
from tensorflow.keras.models import load_model # Using keras from tensorflow is more robust

#  FOCAL LOSS DEFINITION
# This is required to correctly load a model that was compiled with a custom loss.
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function used during training."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: preprocessing, prediction, and probabilities.
    """

    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32, 32)):
        # CRITICAL FIX 2: Pass custom_objects to load_model
        # Use the custom_objects dictionary to define focal_loss upon loading.
        self.model = load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss()}, 
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image):
        """
        Accepts PIL Image, returns processed image tensor.
        CRITICAL FIX: Rescales the loaded image values back to the 0, 1.0, 2.0 range.
        """
        # Convert PIL image to NumPy array and resize
        if isinstance(wafer_image, Image.Image):
            # Convert to grayscale and resize
            wafer_image = wafer_image.convert("L").resize(self.image_size)
            wafer_image = np.array(wafer_image, dtype=np.float32)
        else:
            raise ValueError("Input must be a PIL Image.")

        # Ensure 2D grayscale
        if wafer_image.ndim != 2:
            wafer_image = wafer_image[:, :, 0]

        # --- CRITICAL FIX 3: Revert Scaling ---
        # If max is high (e.g., 255), we must map the non-zero values back to 1.0 and 2.0.
        max_val = wafer_image.max()
        if max_val > 2.0:
            unique_non_zero = np.unique(wafer_image[wafer_image > 0])
            
            # Map the non-zero values back to 1.0 and 2.0
            if len(unique_non_zero) >= 2:
                # Assuming lower is 1.0 (functional) and higher is 2.0 (defective)
                lower_val = np.min(unique_non_zero)
                higher_val = np.max(unique_non_zero)
                
                # Apply the mappings
                wafer_image[wafer_image == lower_val] = 1.0
                wafer_image[wafer_image == higher_val] = 2.0
            
            elif len(unique_non_zero) == 1:
                 # If only one non-zero value, assume binary 0/1 map, map non-zero to 1.0
                 wafer_image[wafer_image > 0] = 1.0
            
            else:
                # If image is just 0s, keep as is
                pass
        
        # Ensure the final array is float32, as expected by the model
        wafer_image = wafer_image.astype(np.float32)

        # Reshape for CNN (Add batch and channel dimensions)
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
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
            # This should rarely happen with argmax but provides robustness
            label = self.le.classes_[np.argmax(preds[:len(self.le.classes_)])] 

        # Probabilities dictionary
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) for i in range(len(self.le.classes_))}
        return label, probs
