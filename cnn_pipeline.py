
import numpy as np
import joblib
from keras.models import load_model
from PIL import Image
import os

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: preprocessing, prediction, and probabilities.
    """

    def __init__(self, model_path: str, label_encoder_path: str = None, image_size=(32, 32)):
        # Load CNN model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN model not found: {model_path}")
        self.model = load_model(model_path, compile=False)  # Keras 3

        # Load LabelEncoder if path provided
        self.le = None
        if label_encoder_path and os.path.exists(label_encoder_path):
            self.le = joblib.load(label_encoder_path)

        self.image_size = image_size

    def preprocess(self, wafer_image):
        """
        Accepts NumPy array or PIL Image, returns processed image tensor.
        Auto-detects if scaling is needed.
        """
        # Convert PIL image to NumPy array and resize
        if isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size, Image.NEAREST)
            wafer_image = np.array(wafer_image)
        elif isinstance(wafer_image, np.ndarray):
            # Resize if shape mismatch
            if wafer_image.shape != self.image_size:
                wafer_image = np.array(Image.fromarray(wafer_image).resize(self.image_size, Image.NEAREST))
        else:
            raise ValueError("Input must be a NumPy array or PIL Image")

        # Ensure 2D grayscale
        if wafer_image.ndim == 3 and wafer_image.shape[2] == 3:
            wafer_image = wafer_image[:, :, 0]

        # Normalize to [0,1] if max > 1
        if wafer_image.max() > 1.0:
            wafer_image = wafer_image / 255.0

        # Reshape for CNN
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
        """
        Returns predicted label (string) and probability dictionary.
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]  # array of probabilities
        pred_class = int(np.argmax(preds))

        # Map to label
        if self.le is not None:
            if pred_class in range(len(self.le.classes_)):
                label = self.le.inverse_transform([pred_class])[0]
            else:
                label = f"Unknown ({pred_class})"
        else:
            label = str(pred_class)

        # Build probabilities dict with human-readable labels
        if self.le is not None:
            probs = {self.le.inverse_transform([i])[0]: float(preds[i]) 
                     for i in range(len(preds))}
        else:
            probs = {str(i): float(preds[i]) for i in range(len(preds))}

        return label, probs