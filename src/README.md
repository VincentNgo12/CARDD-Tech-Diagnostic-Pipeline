# Core Architecture (`/src`)

This directory houses the production-grade Python modules for the CARDD-Tech Diagnostic Pipeline. 

To maintain scalability across diverse research datasets (motors, ventilators, underground cables), this codebase strictly enforces **Object-Oriented Programming (OOP)** principles and **modular decoupling**. 

Code inside `/src` must remain framework-agnostic where possible, acting as a bridge between raw data ingestion, TISEAN-based feature extraction, and our hybrid machine learning backends (`scikit-learn` and `PyTorch`).

---

## 📂 Directory Breakdown & Module Responsibilities

### 1. The Universal Wrapper (`/ingestion`)
This module is the intake valve for the entire pipeline. Because our lab handles vastly different data shapes (e.g., 6-channel high-frequency time-series vs. frequency-domain transfer functions), we utilize an **Adapter Pattern**.

* **`base_loader.py`**: The core Abstract Base Class (ABC). It strictly defines the contract that all future datasets must follow. It prevents fragmented data loading logic by enforcing three mandatory methods:
  * `load_and_clean()`: Handles missing values, NaN imputation, and standard scaling.
  * `get_numpy()`: Outputs `(X, y)` arrays optimized for CPU-bound `scikit-learn` algorithms.
  * `get_tensors()`: Outputs `(X_tensor, y_tensor)` cast to the appropriate `torch.device` (CPU/CUDA) for deep learning.
* **Concrete Loaders (`motor_loader.py`, `cable_loader.py`)**: Inherit from `BaseDataLoader`. They contain the dataset-specific logic (e.g., stripping string timestamps, defining column headers) but output standardized arrays via the parent contract.

### 2. Non-Linear Dynamics & Extraction (`/features`)
This module handles the mathematical transformation of time-series data before it reaches the predictive models. Standard ML models are blind to time; this module gives the data "memory."

* **`delay_embed.py`**: Integrates **TISEAN** concepts (Takens' Theorem). It calculates the optimal time delay ($d$) and embedding dimension ($m$) via Mutual Information and False Nearest Neighbors, folding 1D data streams into multi-dimensional phase-space vectors.
* **`normalizer.py`**: Handles sequence-aware normalization, ensuring that temporal relationships are not destroyed during standard scaling operations.

### 3. Hybrid Diagnostic Models (`/models`)
Contains the actual anomaly detection and classification baselines. The architecture is split into a two-stage hybrid approach to balance deep feature extraction with shallow interpretability.

* **Stage 1 (Anomaly Detectors)**: 
  * `isolation_forest.py`: Shallow, tree-based boundary detection.
  * `oc_svm.py`: One-Class Support Vector Machine for semi-supervised outlier flagging.
  * `oc_nn_pytorch.py`: The Deep One-Class Neural Network baseline. It utilizes a Convolutional Autoencoder (CAE) architecture to compress sequence data and flag high reconstruction errors.
* **Stage 2 (Diagnostic Classifiers)**:
  * Uses cost-sensitive learning algorithms (like weighted Random Forests) and synthetic data generation (`imblearn` SMOTE) to diagnose specific failure modes from highly imbalanced labels.

### 4. Automated Reporting (`/evaluation`)
Separates the evaluation logic from the model training logic to ensure unbiased, standardized benchmarking.

* **`metrics.py`**: Calculates Macro F1-Scores, Recall on minority fault classes, and real-time inference latency (milliseconds per forward pass).
* **`pdf_generator.py`**: Automates the creation of executive summaries, generating Confusion Matrices and Feature Importance bar charts for Explainable AI (XAI) audits.

---

## 🔄 Standard Data Flow (Execution Example)

When calling these modules from a Jupyter Notebook or a deployment script, the data flow must follow this linear topology:

```python
from ingestion.motor_loader import MotorLoader
from features.delay_embed import PhaseSpaceEmbedder
from models.oc_nn_pytorch import DeepOneClass

# 1. Ingestion via Universal Wrapper
loader = MotorLoader("data/raw_motor_signals.csv")
X_raw, _ = loader.get_tensors()

# 2. Temporal Reconstruction
embedder = PhaseSpaceEmbedder(dimension=3, delay=2)
X_embedded = embedder.transform(X_raw)

# 3. Anomaly Detection
model = DeepOneClass(device='cuda')
model.fit(X_embedded)
anomalies = model.predict(X_embedded)