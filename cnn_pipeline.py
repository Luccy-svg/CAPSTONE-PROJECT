
# cnn_pipeline.py

import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

class WaferCNNPipeline:
    """
    A simple CNN pipeline that handles image preprocessing and prediction.
    """

    def __init__(self, model_path: str, label_encoder_path: str, image_size=(26, 26)):
        """
        Initialize pipeline with model and label encoder.
        """
        self.model = load_model(model_path)
        self.le = joblib.load(label_encoder_path)
        self.image_size = image_size

    def preprocess(self, wafer_image):
        """
        Accepts a NumPy array or PIL Image and returns processed image tensor.
        """
        if isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size)
            wafer_image = np.array(wafer_image)
        elif not isinstance(wafer_image, np.ndarray):
            raise ValueError("Input must be a NumPy array or PIL Image")

        # Normalize and reshape
        wafer_image = wafer_image / 255.0
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
        """
        Runs prediction and returns decoded label.
        """
        wafer_processed = self.preprocess(wafer_image)
        pred = np.argmax(self.model.predict(wafer_processed), axis=1)
        label = self.le.inverse_transform(pred)[0]
        return label