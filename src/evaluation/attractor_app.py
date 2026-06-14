"""
Interactive Phase-Space Attractor Animator.

A targeted prototyping GUI built with Streamlit to visually validate optimal 
time-delay embedding configurations discovered via Mutual Information.
"""

import os
import pandas as pd
import numpy as np
import streamlit as strl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json

# ── 1. ENVIRONMENT & CONFIGURATION ──────────────────────────────────────────
strl.set_page_config(page_title="Attractor Unfolding Sandbox", layout="wide")

# Path to pristine baseline continuous training data
DATA_PATH = "/content/drive/MyDrive/CARDD-Tech-Diagnostic-Pipeline/data/LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv"

@strl.cache_data
def load_and_sanitize_data(path: str) -> pd.DataFrame:
    """Loads LBNL data and strips non-continuous variables to prevent geometric collapse."""
    if not os.path.exists(path):
        alternative_path = os.path.abspath(os.path.join("../../", path))
        if os.path.exists(alternative_path):
            path = alternative_path
        else:
            raise FileNotFoundError(f"Could not locate LBNL dataset at: {path}")
            
    df = pd.read_csv(path)
    
    # Drop categorical metadata, discrete step actuators, and flat summer loops
    # INCLUDES THE BINARY AIRFLOW FIX
    cols_to_drop = [
        "Datetime", "FCU_CTRL", "FAN_CTRL",
        "RMCLGSPT", "RMHTGSPT",
        "FCU_CVLV", "FCU_CVLV_DM", "FCU_HVLV", "FCU_HVLV_DM",
        "FCU_CLG_GPM", "FCU_HTG_GPM",
        "FCU_DMPR", "FCU_DMPR_DM", "FCU_SPD", "FCU_WAT",
        "FCU_CLG_EWT", "FCU_CLG_RWT", "FCU_HTG_EWT",
        "FCU_OA_CFM", "FCU_DA_CFM" 
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df.astype(np.float64)


# Define path to your TISEAN output summary
SUMMARY_JSON_PATH = "/content/drive/MyDrive/CARDD-Tech-Diagnostic-Pipeline/tisean_results_lbnl/summary.json"

def get_preset_tau(feature_name: str, json_path: str) -> int:
    """Reads summary.json to find the automated optimal delay for a feature."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 1. Step into the 'parameters' object first
            params = data.get("parameters", {})
            
            # 2. Look for the feature and grab 'd_delay'
            if feature_name in params:
                return int(params[feature_name].get("d_delay", 50))
        except Exception:
            pass # Gracefully ignore parsing issues and fall back
    return 50

# ── 2. DATA INGESTION ────────────────────────────────────────────────────────
try:
    df_clean = load_and_sanitize_data(DATA_PATH)
except Exception as e:
    strl.error(f"Data loading failed: {e}")
    strl.stop()


# ── 3. USER INTERFACE LAYOUT ────────────────────────────────────────────────
strl.title("🔄 Multivariate Phase-Space Unfolding Sandbox")

strl.sidebar.header("Configuration Panel")
selected_feature = strl.sidebar.selectbox("Target Sensor Stream", options=df_clean.columns)

# Visualization Mode Toggle
view_mode = strl.sidebar.radio(
    "Select Geometric View",
    ["2D Phase Space", "3D Attractor", "Poincaré Section (Peak Return Map)"]
)

strl.sidebar.markdown("---")
strl.sidebar.subheader("Time Delay Control (tau)")

# Fetch the automated preset baseline from your JSON summary
preset_tau = get_preset_tau(selected_feature, SUMMARY_JSON_PATH)

# Create a clean, stacked layout for both manual input and sliding
input_col, slider_col = strl.sidebar.columns([1, 2])

with input_col:
    # Numeric Entry Box (changes here update the slider)
    tau_input = strl.number_input(
        "Value",
        min_value=1,
        max_value=1200,
        value=preset_tau,
        step=1
    )

with slider_col:
    # Slider (changes here update the numeric box, defaults to the JSON preset)
    tau_slider = strl.slider(
        "Slide to Adjust", 
        min_value=1, 
        max_value=1200, 
        value=int(tau_input),
        step=1,
        label_visibility="collapsed" # Hides redundant label text for a clean layout
    )

# Establish the final active tau value for calculations
tau = tau_slider


# ── 4. GEOMETRIC STATE CALCULATION ──────────────────────────────────────────
signal = df_clean[selected_feature].values

# 2D Coordinates
x_coords = signal[tau:]
y_coords = signal[:-tau]

# 3D Coordinates
x_3d = signal[2*tau:]
y_3d = signal[tau:-tau]
z_3d = signal[:-2*tau]


# ── 5. LIVE GRAPH RENDERING ─────────────────────────────────────────────────
col1, col2 = strl.columns([1, 1.2])

with col1:
    strl.subheader(f"1D Time-Series Trajectory: `{selected_feature}`")
    fig_time, ax_time = plt.subplots(figsize=(7, 5))
    
    display_slice = 5000 if len(signal) > 5000 else len(signal)
    ax_time.plot(signal[:display_slice], color="#1f77b4", linewidth=1.5, label="Raw Stream")
    
    ax_time.set_xlabel("Time Horizon (Minutes)", fontsize=10)
    ax_time.set_ylabel("Sensor Engineering Units", fontsize=10)
    ax_time.grid(True, linestyle="--", alpha=0.5)
    ax_time.legend(loc="upper right")
    
    strl.pyplot(fig_time)

with col2:
    if view_mode == "2D Phase Space":
        strl.subheader("2D Phase Space Canvas")
        fig_phase, ax_phase = plt.subplots(figsize=(7, 5))
        
        ax_phase.scatter(x_coords, y_coords, alpha=0.15, s=1.5, color="#e377c2", rasterized=True)
        ax_phase.set_xlabel("Current Value: x(t)", fontsize=10)
        ax_phase.set_ylabel(f"Historical Value: x(t - {tau})", fontsize=10)
        ax_phase.grid(True, linestyle="--", alpha=0.5)
        ax_phase.set_xlim(np.min(signal), np.max(signal))
        ax_phase.set_ylim(np.min(signal), np.max(signal))
        
        strl.pyplot(fig_phase)

    elif view_mode == "3D Attractor":
        strl.subheader("3D Phase Space Attractor")
        
        # Prevent browser memory crashes by slicing the first 20,000 points (~14 days)
        PLOT_LIMIT = 20000
        x_render = x_3d[:PLOT_LIMIT]
        y_render = y_3d[:PLOT_LIMIT]
        z_render = z_3d[:PLOT_LIMIT]
        
        # Plotly for smooth interactive 3D rendering
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x_render, y=y_render, z=z_render,
            mode='markers',
            marker=dict(
                size=1.5,
                color=z_render,                # Color mapped to Z-axis depth
                colorscale='Viridis',   
                opacity=0.4
            )
        )])
        
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title="x(t)",
                yaxis_title=f"x(t - {tau})",
                zaxis_title=f"x(t - 2*{tau})"
            ),
            height=500
        )
        
        strl.plotly_chart(fig_3d, use_container_width=True)

    elif view_mode == "Poincaré Section (Peak Return Map)":
        strl.subheader("Poincaré Section: First Return Map")
        
        # Find local maxima (peaks) with a minimum distance to filter micro-noise
        peaks, _ = find_peaks(signal, distance=30) 
        peak_values = signal[peaks]
        
        # P(n) vs P(n+1)
        pn = peak_values[:-1]
        pn_plus_1 = peak_values[1:]
        
        fig_poincare, ax_poincare = plt.subplots(figsize=(7, 5))
        ax_poincare.scatter(pn, pn_plus_1, alpha=0.5, s=10, color="#2ca02c", edgecolor='black', linewidth=0.5)
        
        ax_poincare.set_xlabel("Peak n", fontsize=10)
        ax_poincare.set_ylabel("Peak n+1", fontsize=10)
        ax_poincare.grid(True, linestyle="--", alpha=0.5)
        ax_poincare.set_title("Intersection of the Trajectory Plane", fontsize=11)
        
        strl.pyplot(fig_poincare)