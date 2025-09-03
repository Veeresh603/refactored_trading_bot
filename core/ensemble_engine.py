# core/ensemble_engine.py
import numpy as np

class EnsembleEngine:
    def __init__(self, models, weights=None):
        """
        Ensemble Engine for blending model predictions.

        Args:
            models (list): List of model objects with .predict(data) method
            weights (list, optional): Weight for each model (defaults to equal)
        """
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)

    def predict(self, data):
        """
        Generate an ensemble prediction.

        Args:
            data (dict or np.array): Input features

        Returns:
            int: Signal → +1 (BUY), -1 (SELL), 0 (HOLD)
        """
        preds = [m.predict(data) for m in self.models]
        weighted_preds = np.average(preds, axis=0, weights=self.weights)
        # Thresholding for signals
        if weighted_preds > 0.1:
            return 1
        elif weighted_preds < -0.1:
            return -1
        else:
            return 0
