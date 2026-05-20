import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.ingestion.base_loader import BaseDataLoader


class MotorDataLoader(BaseDataLoader):
    """
    Concrete data loader for the 2023 Induction Motor Fault Dataset.

    This loader processes 6-channel time-series data (3 current, 3 voltage)
    sampled at 10 kHz. It performs domain-specific preprocessing including
    Standard Scaling, which is crucial for neural network convergence and 
    distance-based anomaly detection (like SVM).

    Attributes:
        sensor_cols (List[str]): Names of the 6 electrical signal columns.
        label_col (str): Name of the column containing fault labels.
        df (Optional[pd.DataFrame]): Internal storage for the cleaned data.
        scaler (StandardScaler): Scaler instance used for normalization.
    """

    def __init__(
        self, 
        data_path: str, 
        sensor_cols: List[str] = ["I1", "I2", "I3", "V1", "V2", "V3"],
        label_col: str = "label"
    ) -> None:
        """
        Initializes the motor data loader.

        Args:
            data_path: Path to the CSV dataset.
            sensor_cols: List of column names representing the 6 sensor channels.
            label_col: The target label column name.
        """
        super().__init__(data_path)
        self.sensor_cols = sensor_cols
        self.label_col = label_col
        self.df: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()

    def load_and_clean(self) -> None:
        """
        Loads the motor dataset and performs preprocessing.

        Design Choices:
        1. Forward-fill (ffill): Since this is high-frequency time-series data,
           missing values are best approximated by the previous valid sample
           to maintain temporal continuity.
        2. Standard Scaling: Normalizing to zero mean and unit variance prevents
           features with larger magnitudes (e.g., voltage) from dominating
           those with smaller magnitudes (e.g., current) during model training.
        """
        # Validate file existence
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Motor dataset not found at: {self.data_path}")

        # Load CSV
        self.df = pd.read_csv(self.data_path)

        # Handle missing values: Forward-fill then drop any remaining (e.g., at the start)
        # This preserves the time-series structure better than simple mean imputation.
        self.df[self.sensor_cols] = self.df[self.sensor_cols].ffill().dropna()
        
        # Recalculate df to ensure labels align with dropped rows if any
        self.df = self.df.dropna(subset=self.sensor_cols + [self.label_col])

        # Apply Standard Scaling to sensor data
        # We store the scaler state internally to allow for future inverse transforms
        self.df[self.sensor_cols] = self.scaler.fit_transform(self.df[self.sensor_cols])

    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves features and labels as NumPy arrays for scikit-learn models.

        Returns:
            X: Normalized feature matrix (n_samples, 6).
            y: Label array (n_samples,).
        
        Raises:
            RuntimeError: If load_and_clean() has not been called yet.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_and_clean() first.")

        X = self.df[self.sensor_cols].to_numpy()
        y = self.df[self.label_col].to_numpy()
        return X, y

    def get_tensors(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves features and labels as PyTorch Tensors.

        Args:
            device: Target device for the tensors ('cpu' or 'cuda').

        Returns:
            X_tensor: Float32 tensor of features.
            y_tensor: Long tensor of labels.
        """
        X_np, y_np = self.get_numpy()

        # Convert to float32 for PyTorch, which is standard for weights/activations
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
        # Labels are typically integers for classification/anomaly detection
        y_tensor = torch.tensor(y_np, dtype=torch.long).to(device)

        return X_tensor, y_tensor
