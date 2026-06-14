"""
Interactive Phase-Space Attractor Animator.

A targeted prototyping GUI built with Streamlit to visually validate optimal 
time-delay embedding configurations discovered via Mutual Information.
Now fully integrated with TISEAN's native `delay` C-binary.
"""

import os
import subprocess
import json
import pandas as pd
import numpy as np
import streamlit as strl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks

# ── 1. ENVIRONMENT & CONFIGURATION ──────────────────────────────────────────
strl.set_page_config(page_title="Attractor Unfolding Sandbox", layout="wide")

DATA_PATH = "/content/drive/MyDrive/CARDD-Tech-Diagnostic-Pipeline/data/LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv"
SUMMARY_JSON_PATH = "/content/drive/MyDrive/CARDD-Tech-Diagnostic-Pipeline/tisean_results_lbnl/summary.json"

# Define TISEAN executable path
PROJECT_ROOT = '/content/drive/MyDrive/CARDD-Tech-Diagnostic-Pipeline'
_TISEAN_DIR = f"{PROJECT_ROOT}/Tisean_3.0.1/source_c"
DELAY_EXE = os.path.abspath(os.path.join(_TISEAN_DIR, "delay"))

@strl.cache_data
def load_and_sanitize_data(path: str) -> pd.DataFrame:
    """Loads LBNL data and strips non-continuous variables."""
    if not os.path.exists(path):
        alternative_path = os.path.abspath(os.path.join("../../", path))
        path = alternative_path if os.path.exists(alternative_path) else path
            
    df = pd.read_csv(path)
    
    cols_to_drop = [
        "Datetime", "FCU_CTRL", "FAN_CTRL", "RMCLGSPT", "RMHTGSPT",
        "FCU_CVLV", "FCU_CVLV_DM", "FCU_HVLV", "FCU_HVLV_DM",
        "FCU_CLG_GPM", "FCU_HTG_GPM", "FCU_DMPR", "FCU_DMPR_DM", 
        "FCU_SPD", "FCU_WAT", "FCU_CLG_EWT", "FCU_CLG_RWT", 
        "FCU_HTG_EWT", "FCU_OA_CFM", "FCU_DA_CFM" 
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df.astype(np.float64)

def get_preset_tau(feature_name: str, json_path: str) -> int:
    """Reads summary.json to find the automated optimal delay."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            params = data.get("parameters", {})
            if feature_name in params:
                return int(params[feature_name].get("d_delay", 50))
        except Exception:
            pass
    return 50

# ── 2. TISEAN DELAY WRAPPER ─────────────────────────────────────────────────

# The underscore in `_signal` tells Streamlit NOT to hash the massive array, 
# preventing memory overflow. It caches purely based on feature, tau, and m.
@strl.cache_data(show_spinner=False)
def generate_tisean_vectors(_signal: np.ndarray, feature: str, tau: int, m: int) -> np.ndarray:
    """Executes the TISEAN delay C-binary and retrieves the vector matrix."""
    if not os.path.isfile(DELAY_EXE):
        strl.error(f"TISEAN delay binary not found at: {DELAY_EXE}")
        strl.stop()

    in_file = f"temp_delay_in_{feature}_m{m}_d{tau}.txt"
    out_file = f"temp_delay_out_{feature}_m{m}_d{tau}.txt"
    
    # Write 1D data for TISEAN
    np.savetxt(in_file, _signal, fmt="%.8f")
    
    # Run: delay -m <dim> -d <tau> input.txt -o output.txt
    subprocess.run(
        [DELAY_EXE, "-m", str(m), "-d", str(tau), in_file, "-o", out_file],
        capture_output=True
    )
    
    # Load the resulting matrix
    if os.path.exists(out_file):
        matrix = np.loadtxt(out_file)
    else:
        matrix = np.zeros((1, m)) # Fallback if binary fails
        
    # Cleanup temp files
    if os.path.exists(in_file): os.remove(in_file)
    if os.path.exists(out_file): os.remove(out_file)
        
    return matrix


# ── 3. DATA INGESTION & UI ──────────────────────────────────────────────────
try:
    df_clean = load_and_sanitize_data(DATA_PATH)
except Exception as e:
    strl.error(f"Data loading failed: {e}")
    strl.stop()

strl.title("🔄 Multivariate Phase-Space Unfolding Sandbox")

strl.sidebar.header("Configuration Panel")
selected_feature = strl.sidebar.selectbox("Target Sensor Stream", options=df_clean.columns)

view_mode = strl.sidebar.radio(
    "Select Geometric View",
    ["2D Phase Space", "3D Attractor", "Poincaré Section (Peak Return Map)"]
)

strl.sidebar.markdown("---")
strl.sidebar.subheader("Time Delay Control (tau)")

preset_tau = get_preset_tau(selected_feature, SUMMARY_JSON_PATH)
input_col, slider_col = strl.sidebar.columns([1, 2])

with input_col:
    tau_input = strl.number_input("Value", min_value=1, max_value=1200, value=preset_tau, step=1)
with slider_col:
    tau_slider = strl.slider("Slide to Adjust", min_value=1, max_value=1200, value=int(tau_input), step=1, label_visibility="collapsed")

tau = tau_slider
signal = df_clean[selected_feature].values


# ── 4. LIVE GRAPH RENDERING ─────────────────────────────────────────────────
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
        strl.subheader("2D Phase Space Canvas (TISEAN)")
        
        # Pull 2D vectors from TISEAN
        delay_matrix = generate_tisean_vectors(signal, selected_feature, tau, m=2)
        
        # --- THE FIX: Sub-sampling the array ---
        # "::20" means start at the beginning, go to the end, but only take every 20th point.
        # This breaks the "connected line" illusion by forcing gaps between the dots.
        STEP = 200 
        x_coords = delay_matrix[::STEP, 0]
        y_coords = delay_matrix[::STEP, 1]
        
        fig_phase, ax_phase = plt.subplots(figsize=(7, 5))
        
        # Because we have way fewer points now, we can increase the opacity (alpha) 
        # and size (s) so the individual dots look crisp and distinct.
        ax_phase.scatter(x_coords, y_coords, alpha=0.6, s=5.0, color="#e377c2", rasterized=True)
        
        ax_phase.set_xlabel("Current Value: x(t)", fontsize=10)
        ax_phase.set_ylabel(f"Historical Value: x(t - {tau})", fontsize=10)
        ax_phase.grid(True, linestyle="--", alpha=0.5)
        ax_phase.set_xlim(np.min(signal), np.max(signal))
        ax_phase.set_ylim(np.min(signal), np.max(signal))
        
        strl.pyplot(fig_phase)

    elif view_mode == "3D Attractor":
        strl.subheader("3D Phase Space Attractor (TISEAN)")
        
        delay_matrix = generate_tisean_vectors(signal, selected_feature, tau, m=3)
        
        # Slice it down to 20,000 points, but take every 10th point
        PLOT_LIMIT = 20000
        STEP_3D = 10
        x_render = delay_matrix[:PLOT_LIMIT:STEP_3D, 0]
        y_render = delay_matrix[:PLOT_LIMIT:STEP_3D, 1]
        z_render = delay_matrix[:PLOT_LIMIT:STEP_3D, 2]
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x_render, y=y_render, z=z_render,
            mode='markers',
            marker=dict(
                size=2.0,
                color=z_render, 
                colorscale='Viridis',   
                opacity=0.6
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
        
        peaks, _ = find_peaks(signal, distance=30) 
        peak_values = signal[peaks]
        
        pn = peak_values[:-1]
        pn_plus_1 = peak_values[1:]
        
        fig_poincare, ax_poincare = plt.subplots(figsize=(7, 5))
        # Already set as pure scatter
        ax_poincare.scatter(pn, pn_plus_1, alpha=0.5, s=10, color="#2ca02c", edgecolor='black', linewidth=0.5)
        
        ax_poincare.set_xlabel("Peak n", fontsize=10)
        ax_poincare.set_ylabel("Peak n+1", fontsize=10)
        ax_poincare.grid(True, linestyle="--", alpha=0.5)
        ax_poincare.set_title("Intersection of the Trajectory Plane", fontsize=11)
        
        strl.pyplot(fig_poincare)