
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model 

# -------------------- FOCAL LOSS DEFINITION -------------------- #
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function used during training."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# -------------------- CNN PIPELINE CLASS -------------------- #
class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: handles model loading, preprocessing, 
    and prediction for wafer map images or numpy arrays.
    """
    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32, 32)):
        self.model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()},
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_input) -> np.ndarray:
        """
        Accepts a PIL Image or 2D NumPy array, returns processed image tensor
        suitable for CNN prediction.
        """
        # Convert numpy array to PIL Image if necessary
        if isinstance(wafer_input, np.ndarray):
            wafer_input = Image.fromarray(wafer_input.astype(np.uint8))
        elif not isinstance(wafer_input, Image.Image):
            raise ValueError("Input must be a PIL Image or 2D NumPy array.")

        # Convert to grayscale and resize
        wafer_image = wafer_input.convert("L").resize(self.image_size, Image.NEAREST)
        wafer_array = np.array(wafer_image, dtype=np.float32)

        # Rescale values: map wafer values to 0,1,2 (robust handling)
        max_val = wafer_array.max()
        if max_val > 2.0:
            unique_non_zero = np.unique(wafer_array[wafer_array > 0])
            if len(unique_non_zero) >= 2:
                sorted_unique = np.sort(unique_non_zero)
                split_point = (sorted_unique[-1] + sorted_unique[-2]) / 2.0
                wafer_array[wafer_array > split_point] = 2.0
                wafer_array[(wafer_array > 0.0) & (wafer_array <= split_point)] = 1.0
            elif len(unique_non_zero) == 1:
                wafer_array[wafer_array > 0.0] = 1.0

        # Add batch and channel dimensions: [1, H, W, 1]
        wafer_array = wafer_array.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_array

    def predict(self, wafer_input):
        """
        Returns predicted label and probabilities dictionary for known classes.
        Handles model outputs that exceed label encoder size.
        """
        x = self.preprocess(wafer_input)
        preds = self.model.predict(x, verbose=0)[0]  # shape: (num_model_outputs,)

        # Safe predicted class index
        pred_class_idx = int(np.argmax(preds))
        num_labels = len(self.le.classes_)
        if pred_class_idx >= num_labels:
            pred_class_idx = num_labels - 1  # fallback to last known label

        label = self.le.inverse_transform([pred_class_idx])[0]

        # Probabilities dictionary safely
        probs = {}
        for i in range(len(preds)):
            if i < num_labels:
                probs[self.le.inverse_transform([i])[0]] = float(preds[i])
            else:
                probs[f"Unknown_{i}"] = float(preds[i])

        return label, probs