
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------- FOCAL LOSS -------------------- #
def focal_loss(gamma=2., alpha=0.25):
    """Focal loss function for loading CNN with custom loss."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1. - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# -------------------- Wafer CNN Pipeline -------------------- #
class WaferCNNPipeline:
    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32,32)):
        # Load model with custom loss
        self.model = load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()},
            compile=False
        )
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image) -> np.ndarray:
        """
        Preprocess wafer image (PIL or NumPy array) for CNN input.
        Converts any image to 32x32 grayscale with 0,1,2 mapping.
        """
        # Handle NumPy arrays
        if isinstance(wafer_image, np.ndarray):
            wafer_image = Image.fromarray(wafer_image.astype(np.uint8))

        if not isinstance(wafer_image, Image.Image):
            raise ValueError("Input must be PIL Image or NumPy array.")

        # Convert to grayscale and resize
        wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
        wafer_array = np.array(wafer_image, dtype=np.float32)

        # Map pixel values to 0,1,2 (background / minor defect / defect)
        wafer_array[wafer_array > 128] = 2
        wafer_array[(wafer_array > 0) & (wafer_array <= 128)] = 1
        wafer_array[wafer_array == 0] = 0

        # Reshape for CNN: [1, H, W, 1]
        wafer_array = wafer_array.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_array

    def predict(self, wafer_image):
        """
        Returns predicted label and probabilities dictionary.
        Handles alignment with label encoder.
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]

        # Predicted class
        pred_class = int(np.argmax(preds))
        if pred_class < len(self.le.classes_):
            label = self.le.inverse_transform([pred_class])[0]
        else:
            # Safety fallback
            label = self.le.classes_[np.argmax(preds[:len(self.le.classes_)])]

        # Probabilities dictionary
        probs = {self.le.inverse_transform([i])[0]: float(preds[i]) 
                 for i in range(len(self.le.classes_))}
        return label, probs