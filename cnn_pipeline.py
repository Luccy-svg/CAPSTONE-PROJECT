
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------- FOCAL LOSS -------------------- #
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss used during training."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# -------------------- CNN PIPELINE -------------------- #
class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3:
    - Handles preprocessing
    - Prediction
    - Class weight adjustment
    """
    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32,32), class_weights=None):
        self.model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()},
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size
        self.class_weights = class_weights  # Optional class weight dict

    def preprocess(self, wafer_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to CNN-ready array [1,H,W,1] with values 0,1,2."""
        if not isinstance(wafer_image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
        arr = np.array(wafer_image, dtype=np.float32)

        # Rescale to 0,1,2
        max_val = arr.max()
        if max_val > 2.0:
            unique_non_zero = np.unique(arr[arr>0])
            if len(unique_non_zero) >= 2:
                sorted_unique = np.sort(unique_non_zero)
                split_point = (sorted_unique[-1] + sorted_unique[-2]) / 2.0
                arr[arr>split_point] = 2.0
                arr[(arr>0) & (arr<=split_point)] = 1.0
            elif len(unique_non_zero) == 1:
                arr[arr>0] = 1.0
        return arr.reshape(1, self.image_size[0], self.image_size[1], 1)

    def predict(self, wafer_image: Image.Image):
        """
        Returns:
        - predicted label
        - probabilities dict adjusted with class weights if provided
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]

        # Apply class weights if provided
        if self.class_weights:
            weight_vector = np.array([self.class_weights.get(i, 1.0) for i in range(len(preds))])
            preds = preds * weight_vector
            preds = preds / preds.sum()  # Re-normalize

        pred_idx = int(np.argmax(preds))
        label = self.le.inverse_transform([pred_idx])[0]

        # Probabilities dict
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) for i in range(len(self.le.classes_))}
        return label, probs