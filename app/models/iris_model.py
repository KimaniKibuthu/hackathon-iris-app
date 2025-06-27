"""
Iris classification model handling.

This module provides functionality for loading and using the pre-trained
iris classification model.
"""

import joblib
import logging
from pathlib import Path
from typing import List, Union
import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError

logger = logging.getLogger(__name__)


class IrisClassifier:
    """Iris flower classifier using a pre-trained SVC model."""
    
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize the iris classifier.
        
        Args:
            model_path: Path to the pre-trained model file.
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.class_names = ['setosa', 'versicolor', 'virginica']
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained model from disk."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Could not load model: {e}")
    
    def predict(self, features: List[float]) -> dict:
        """
        Predict the iris species based on input features.
        
        Args:
            features: List of four float values representing sepal length,
                     sepal width, petal length, and petal width.
        
        Returns:
            Dictionary containing prediction and confidence scores.
        
        Raises:
            PredictionError: If prediction fails.
        """
        try:
            # Validate input
            if len(features) != 4:
                raise PredictionError("Expected exactly 4 features")
            
            if not all(isinstance(f, (int, float)) and f >= 0 for f in features):
                raise PredictionError("All features must be non-negative numbers")
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_array)[0]
                probabilities = {
                    name: float(prob) 
                    for name, prob in zip(self.class_names, proba)
                }
                confidence = float(max(proba))
            else:
                # For models without predict_proba, use decision function or default
                confidence = 1.0
                probabilities = {name: 0.0 for name in self.class_names}
                probabilities[self.class_names[prediction]] = 1.0
            
            # Prepare response
            result = {
                "prediction": self.class_names[prediction],
                "confidence": confidence,
                "probabilities": probabilities
            }
            
            logger.info(f"Prediction made: {result['prediction']} with confidence {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}")


# Global classifier instance
classifier = IrisClassifier()
