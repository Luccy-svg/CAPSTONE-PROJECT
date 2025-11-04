
import numpy as np
import joblib
from keras.models import load_model
from PIL import Image

# Mapping your defect types (matches your notebook)
mapping_type = {
    'Center':0,
    'Donut':1,
    'Edge-Loc':2,
    'Edge-Ring':3,
    'Loc':4,
    'Random':5,
    'Scratch':6,
    'none':7,
    '[0 0]':7,
    'Unknown':7
}

# Reverse mapping: numeric -> name
reverse_mapping = {v:k for k,v in mapping_type.items()}

class WaferCNNPipeline:
    """
    CNN pipeline for Keras 3: preprocessing, prediction, and probabilities.
    Handles unseen labels safely.
    """

    def __init__(self, model_path: str, label_encoder_path: str = None, image_size=(32, 32)):
        self.model = load_model(model_path, compile=False)
        self.image_size = image_size
        
        # Load label encoder if available; otherwise fallback to mapping_type
        if label_encoder_path:
            self.le = joblib.load(label_encoder_path)
            self.use_encoder = True
        else:
            self.le = None
            self.use_encoder = False

    def preprocess(self, wafer_image):
        """Converts a wafer image to model-ready input."""
        if isinstance(wafer_image, Image.Image):
            wafer_image = wafer_image.convert("L").resize(self.image_size)
            wafer_image = np.array(wafer_image)
        elif isinstance(wafer_image, np.ndarray):
            if wafer_image.shape != self.image_size:
                wafer_image = np.array(Image.fromarray(wafer_image).resize(self.image_size))
        else:
            raise ValueError("Input must be a NumPy array or PIL Image")

        if wafer_image.ndim != 2:
            wafer_image = wafer_image[:, :, 0]

        wafer_image = wafer_image / 255.0
        wafer_image = wafer_image.reshape(1, self.image_size[0], self.image_size[1], 1)
        return wafer_image

    def predict(self, wafer_image):
        """
        Returns predicted label and probabilities dictionary.
        Safely maps unseen classes to 'Unknown'.
        """
        x = self.preprocess(wafer_image)
        preds = self.model.predict(x, verbose=0)[0]  # array of probabilities
        pred_class = int(np.argmax(preds))

        # Use label encoder if available
        if self.use_encoder:
            if pred_class < len(self.le.classes_):
                label = self.le.inverse_transform([pred_class])[0]
            else:
                label = 'Unknown'
        else:
            # Fallback: map numeric index to defect name using reverse_mapping
            label = reverse_mapping.get(pred_class, 'Unknown')

        # Probabilities as a dict (map names to probs)
        if self.use_encoder:
            prob_dict = {self.le.classes_[i] if i < len(self.le.classes_) else 'Unknown': float(preds[i])
                         for i in range(len(preds))}
        else:
            prob_dict = {reverse_mapping.get(i, 'Unknown'): float(preds[i]) for i in range(len(preds))}

        return label, prob_dict