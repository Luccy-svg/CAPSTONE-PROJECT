
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: preprocessing, prediction, and probabilities.
    """

    def __init__(self, model_path: str, label_encoder_path: str, image_size=(32, 32)):
        self.model = load_model(model_path, compile=False)  # Keras 3
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image):
        """
        Accepts NumPy array or PIL Image, returns processed image tensor.
        """
        # Convert PIL image to NumPy array
        if isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size)
            wafer_image = np.array(wafer_image)
        elif isinstance(wafer_image, np.ndarray):
            if wafer_image.shape != self.image_size:
                wafer_image = np.array(Image.fromarray(wafer_image).resize(self.image_size))
        else:
            raise ValueError("Input must be a NumPy array or PIL Image")

        # Ensure 2D
        if wafer_image.ndim != 2:
            wafer_image = wafer_image[:, :, 0]

        wafer_image = wafer_image / 255.0
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
        """
        Returns predicted label and probabilities dictionary.
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]  # probabilities
        pred_class = np.argmax(preds)
        label = self.le.inverse_transform([pred_class])[0]
        probs = dict(zip(self.le.classes_, preds))
        return label, probs