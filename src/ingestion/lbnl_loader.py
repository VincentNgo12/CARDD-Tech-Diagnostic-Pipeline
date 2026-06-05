"""
LBNL HVAC Fault Detection Data Loader.

This module provides a specialized implementation of the BaseDataLoader 
for the LBNL (Lawrence Berkeley National Laboratory) HVAC dataset. 
It handles continuous sensor streams and prepares them for 
Multivariate Phase-Space Reconstruction.

Architecture Path Mapping:
- Input: Raw CSV/JSON from LBNL continuous sensor streams (data/raw/lbnl/)
- Output: Standardized NumPy arrays or PyTorch Tensors for delay embedding.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.ingestion.base_loader import BaseDataLoader


class LBNLLoader(BaseDataLoader):
    """
    Data loader for the LBNL continuous sensor stream dataset.

    This class implements the ingestion logic for HVAC fault detection data,
    providing methods to load, clean, and format data for both classical 
    machine learning and deep learning models.

    Attributes:
        data_path (str): Path to the LBNL data directory or file.
        df (Optional[pd.DataFrame]): Internal storage for the loaded DataFrame.
    """

    def __init__(self, data_path: str, label: int) -> None:
        """
        Initializes the LBNL loader.

        Args:
            data_path: Path to the LBNL dataset.
            label: Integer class label (e.g., 0 for Healthy, 1 for Faulty).
        """
        super().__init__(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.label = label

    def load_and_clean(self) -> None:
        """
        Loads raw LBNL data and performs domain-specific cleaning.

        This method replicates the extraction logic from Andy's TISEAN wrapper,
        stripping non-numeric and discrete actuator columns to align with 
        the expected feature list.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"LBNL data not found at: {self.data_path}")
        
        # Load the CSV file
        self.df = pd.read_csv(self.data_path)
        
        # Aggressively strip non-numeric and discrete actuator columns
        cols_to_drop = ["Datetime", "FCU_CTRL", "FAN_CTRL"]
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns])
        
        # Ensure data is numeric
        self.df = self.df.astype(np.float64)
        
        print(f"DEBUG: Loaded and cleaned LBNL data from {self.data_path}. "
              f"Remaining features: {len(self.df.columns)}")

    def get_numpy(self, summary_json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the multivariate phase-space embedding matrix.

        Args:
            summary_json_path: Path to the TISEAN parameter discovery output (summary.json).

        Returns:
            A tuple (X, y) where X is the constructed 2D NumPy embedding matrix 
            and y is a 1D NumPy array of identical length filled with self.label.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean() first.")
        
        import json
        with open(summary_json_path, "r") as f:
            summary = json.load(f)
        
        # Extract discovered optimal delay (d) and embedding dimension (m)
        d_vals = summary["d_optimal"]
        m_vals = summary["m_optimal"]
        
        # Calculate individual lookback windows and the global 'Max Lookback'
        # Lookback = (m - 1) * d
        lookbacks = [(m - 1) * d for m, d in zip(m_vals, d_vals)]
        max_lookback = int(max(lookbacks))
        
        data = self.df.values
        N, num_features = data.shape
        
        # Total number of samples in the reconstructed phase-space
        num_samples = N - max_lookback
        total_cols = sum(m_vals)
        
        # Pre-allocate embedding matrix
        X = np.zeros((num_samples, total_cols), dtype=np.float64)
        
        # Construct the multivariate phase-space embedding matrix using 
        # a high-performance vectorized NumPy slicing loop.
        col_offset = 0
        for i in range(num_features):
            m = m_vals[i]
            d = d_vals[i]
            
            # For each index t from max_lookback to N, horizontally stack 
            # the history vectors: [x_i(t), x_i(t - d_i), x_i(t - 2*d_i), ...]
            for j in range(m):
                lag = j * d
                # Extract the history vector across the entire valid range
                X[:, col_offset] = data[max_lookback - lag : N - lag, i]
                col_offset += 1
                
        # y is a 1D array filled entirely with self.label
        y = np.full((num_samples,), self.label, dtype=np.int64)
        
        return X, y

    def get_tensors(self, summary_json_path: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts the processed LBNL data into PyTorch Tensors.

        Args:
            summary_json_path: Path to the TISEAN summary.json.
            device: Target device for tensors (e.g., 'cpu', 'cuda').

        Returns:
            A tuple (X_tensor, y_tensor) on the specified device.
        """
        X_np, y_np = self.get_numpy(summary_json_path)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_np, dtype=torch.long).to(device)
        return X_tensor, y_tensor
