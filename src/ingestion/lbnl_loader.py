import json
import numpy as np
import pandas as pd
import os

def load_tisean_params(json_path):
    """Loads the optimal delay (d) and dimension (m) parameters."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config["parameters"]

def get_max_history(params_dict):
    """Calculates the absolute longest lookback window needed for alignment."""
    return max((s["m_dimension"] - 1) * s["d_delay"] for s in params_dict.values())

def build_trajectory_matrix(csv_path, params_dict, max_history, step=1):
    """
    Ingests a CSV and transforms the continuous time-series columns into 
    a single synchronized, delay-embedded phase-space matrix.
    """
    df = pd.read_csv(csv_path)
    N_total = len(df) - max_history
    embedded_features = []
    
    for feature, settings in params_dict.items():
        # Safety check: ensure the sensor actually exists in this specific fault file
        if feature not in df.columns:
            raise ValueError(f"CRITICAL: Missing sensor '{feature}' in {os.path.basename(csv_path)}")
            
        signal = df[feature].values
        d = settings["d_delay"]
        m = settings["m_dimension"]
        
        feature_matrix = np.zeros((N_total, m))
        
        # Fast memory-level slicing (Vectorized)
        for i in range(m):
            lag = i * d
            start_idx = max_history - lag
            end_idx = len(signal) - lag
            feature_matrix[:, i] = signal[start_idx:end_idx]
            
        embedded_features.append(feature_matrix)

    # Horizontally stack all feature matrices (e.g., 11 sensors * 5 dims = 54 columns)
    full_matrix = np.hstack(embedded_features)
    
    # Apply STSP Decimation (Theiler Step)
    return full_matrix[::step]