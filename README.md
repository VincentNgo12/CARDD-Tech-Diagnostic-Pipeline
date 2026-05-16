# CARDD-Tech Diagnostic Pipeline
**Centre for Applied Research in Defence and Dual-use Technologies (CARDD-Tech) - Cyber/AI Theme** **University of Alberta**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This repository contains a modular, hardware-agnostic machine learning pipeline for condition monitoring and semi-supervised anomaly detection. 

Developed under the CARDD-Tech Cyber/AI theme, this framework is designed to ingest multi-domain sensor data—ranging from high-frequency time-series motor vibrations to frequency-domain underground cable telemetry—and route it through a standardized hybrid ML architecture.

The core pipeline aims to solve the **Accuracy-Interpretability Tradeoff** by leveraging deep learning (1D-CNNs, Autoencoders) for complex feature extraction, and shallow algorithms (Random Forests, Isolation Forests) for transparent, auditable fault isolation.

---

## 🏗️ System Architecture

The pipeline is separated into three distinct execution stages to ensure modularity and scalability across different research datasets:

### 1. Universal Data Ingestion Wrapper (`/src/ingestion`)
A unified interface (`base_loader.py`) designed to handle highly imbalanced datasets with differing input shapes. Standardizes raw CSV/JSON inputs into PyTorch Tensors and NumPy arrays for downstream processing.
* **Supported domains:** Small induction motors, computational pipeline monitoring (CPM), and XLPE power cable frequency responses.

### 2. Feature Extraction & Non-Linear Dynamics (`/src/features`)
Implements time-series reconstruction using **TISEAN**. Applies Takens' Theorem to calculate optimal delay ($d$) and embedding dimensions ($m$), converting 1D sensor streams into multi-dimensional phase space representations (Delay Embeddings).

### 3. Hybrid Diagnostic Models (`/src/models`)
A suite of baseline algorithms for benchmarking and fault detection:
* **Stage 1 (Anomaly Detection):** Isolation Forests and One-Class SVMs for unsupervised/semi-supervised fault flagging.
* **Stage 2 (Diagnostic Classification):** Cost-sensitive Random Forests and SMOTE-enhanced classifiers to address extreme class imbalance.
* **Deep Baselines:** PyTorch implementations of Convolutional Autoencoders (CAE) and One-Class Neural Networks (OC-NN).

---

## 📂 Repository Structure

```text
cardd-tech-diagnostic-pipeline/
├── data/                    # Local raw datasets (Ignored by Git)
├── notebooks/               # Scratchpads & Exploratory Data Analysis (EDA)
├── src/                     # Core production modules
│   ├── ingestion/           # Data loaders and standardization wrappers
│   ├── features/            # TISEAN integration and temporal embedding
│   ├── models/              # Scikit-learn and PyTorch model definitions
│   └── evaluation/          # Automated PDF reporting and metric calculation
├── tests/                   # Unit testing framework
├── .gitignore               # Security and environment exclusions
├── README.md                # Project documentation
└── requirements.txt         # Dependency management