"""
Supervised Random Forest Classifier for Diagnostic Classification.

This module houses the Random Forest pipeline for supervised classification 
of HVAC faults. It is designed to consume features extracted via 
Phase Space Reconstruction (PSR) or direct sensor streams.

Architecture Path Mapping:
- Input: PSR Delay Embeddings from `src/features/delay_embed.py`
- Output: Fault classification labels and probability estimates.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import Optional, Any


class HVACRandomForest(BaseEstimator, ClassifierMixin):
    """
    Random Forest wrapper for HVAC fault diagnosis.

    Leverages scikit-learn's RandomForestClassifier to provide an auditable 
    and interpretable diagnostic layer in the pipeline.

    Attributes:
        model (RandomForestClassifier): The underlying scikit-learn model.
    """

    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: Optional[int] = None, 
        random_state: int = 42
    ) -> None:
        """
        Initializes the HVAC Random Forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            random_state: Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HVACRandomForest":
        """
        Fits the Random Forest model to the provided data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).

        Returns:
            The fitted model instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts fault classes for the given input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the given input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Class probabilities.
        """
        return self.model.predict_proba(X)
