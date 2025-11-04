
import numpy as np
import joblib
from keras.models import load_model
from PIL import Image

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: preprocessing, prediction, and probabilities.
    """

    # Correct mapping of CNN output indices to defect labels
    DEFECT_MAP = {
        0: 'No Defect',
        1: 'Center',
        2: 'Donut',
        3: 'Edge-Ring',
        4: 'Scratch',
        5: 'Near-full',
        6: 'Random',
        7: 'Local (Loc)',
        8: 'Cluster'
    }

    def __init__(self, model_path: str, label_encoder_path: str = None, image_size=(32, 32)):
        self.model = load_model(model_path, compile=False)  # Keras 3
        self.image_size = image_size
        # Optional label encoder for compatibility, not required here
        self.le = None
        if label_encoder_path:
            self.le = joblib.load(label_encoder_path)

    def preprocess(self, wafer_image):
        """
        Accepts NumPy array or PIL Image, returns processed image tensor.
        """
        # Convert PIL image to NumPy array and resize
        if isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size)
            wafer_image = np.array(wafer_image)
        elif isinstance(wafer_image, np.ndarray):
            # Resize if not the target shape
            if wafer_image.shape != self.image_size:
                wafer_image = np.array(Image.fromarray(wafer_image).resize(self.image_size))
        else:
            raise ValueError("Input must be a NumPy array or PIL Image")

        # Ensure 2D grayscale
        if wafer_image.ndim != 2:
            wafer_image = wafer_image[:, :, 0]

        # Normalize and reshape for CNN
        wafer_image = wafer_image.astype('float32') / 255.0
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
        """
        Returns predicted label and probabilities dictionary.
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]  # softmax probabilities
        pred_class = int(np.argmax(preds))
        label = self.DEFECT_MAP.get(pred_class, f"Unknown ({pred_class})")
        probs = {self.DEFECT_MAP.get(i, f"Unknown ({i})"): float(prob) for i, prob in enumerate(preds)}
        return label, probs