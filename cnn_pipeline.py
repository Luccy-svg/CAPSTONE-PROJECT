
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# -------------------- FOCAL LOSS -------------------- #
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# -------------------- CNN PIPELINE -------------------- #
class WaferCNNPipeline:
    """Handles preprocessing, prediction, and class weight adjustment for Keras 3 CNN"""
    
    def __init__(self, model_path, label_encoder_path, image_size=(32,32), class_weights=None):
        self.model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()},
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size
        self.class_weights = class_weights

    def preprocess(self, wafer_image):
        """Convert input to CNN-ready array [1,H,W,1], normalized 0-1"""
        if isinstance(wafer_image, np.ndarray):
            arr = wafer_image.astype(np.float32)
        elif isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
            arr = np.array(wafer_image, dtype=np.float32)
        else:
            raise ValueError("Input must be PIL Image or numpy array.")
        
        # Normalize
        if arr.max() > 1.0:
            arr /= 255.0

        return arr.reshape(1, self.image_size[0], self.image_size[1], 1)

    def predict(self, wafer_image):
        """Returns predicted label and class probabilities"""
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]

        # Adjust for class weights if provided
        if self.class_weights:
            weight_vector = np.array([self.class_weights.get(i, 1.0) for i in range(len(preds))])
            preds *= weight_vector
            preds /= preds.sum()

        pred_idx = int(np.argmax(preds))
        label = self.le.inverse_transform([pred_idx])[0]

        # Probabilities dict
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) for i in range(len(self.le.classes_))}
        return label, probs