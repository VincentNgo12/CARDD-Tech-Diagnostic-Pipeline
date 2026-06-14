"""
TISEAN parameter discovery for LBNL FDD research - Refactored Edition.

Runs Mutual Information (MI) and False Nearest Neighbors (FNN) on each sensor
feature using the healthy (normal) dataset to find:
  d  — optimal time delay per sensor
  m  — optimal embedding dimension per sensor

Integrated with robust MI local minimum finder, Matplotlib visualization,
advanced CLI management, and input sanitization.
"""

import argparse
import json
import os
import subprocess
import time
from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default configuration constants
_TISEAN_DIR = os.path.join(
    os.path.dirname(__file__),
    "./Tisean_3.0.1/source_c",
)
MUTUAL_EXE = os.path.abspath(os.path.join(_TISEAN_DIR, "mutual"))
FNN_EXE = os.path.abspath(os.path.join(_TISEAN_DIR, "false_nearest"))

DEFAULT_MAX_DELAY = 50
MAX_EMB_DIM = 20
DEFAULT_MAX_ROWS = 200_000 # cap rows fed to TISEAN FNN 
FNN_PLATEAU_DELTA = 0.010  # Delta threshold to determine if the FNN plot plateaued


# ── Utility Functions ────────────────────────────────────────────────────────

def smooth_signal(x: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply a simple moving average smoothing filter to a 1D array.
    
    Args:
        x: Input signal array.
        window_size: Size of the smoothing window.
        
    Returns:
        Smoothed signal array.
    """
    if window_size < 2:
        return x
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')


def plot_mi_curve(
    delays: np.ndarray, 
    mi_raw: np.ndarray, 
    mi_smoothed: np.ndarray, 
    d_opt: int, 
    tag: str, 
    plots_dir: str
) -> None:
    """
    Generate and save a professional plot of the MI curve.
    
    Args:
        delays: Array of time delays.
        mi_raw: Raw mutual information values.
        mi_smoothed: Smoothed mutual information values.
        d_opt: Automatically discovered optimal delay.
        tag: Feature name tag for the filename.
        plots_dir: Directory where plots should be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(delays, mi_raw, alpha=0.3, label='Raw MI', color='gray')
    plt.plot(delays, mi_smoothed, label='Smoothed MI', color='blue', linewidth=2)
    plt.axvline(x=d_opt, color='red', linestyle='--', label=f'Optimal Delay (d={d_opt})')
    
    plt.title(f'Mutual Information vs. Time Delay (tau) - {tag}', fontsize=14)
    plt.xlabel('Time Delay (tau)', fontsize=12)
    plt.ylabel('Mutual Information Score', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plot_filename = os.path.join(plots_dir, f"{tag}_mi_curve.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()


def plot_fnn_sweep(
    dims: np.ndarray, 
    fnn_results: dict, 
    m_opt: int, 
    tag: str, 
    plots_dir: str
) -> None:
    """
    Plots the FNN curves for multiple ratio factors (f) to visualize the invariant elbow.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {2.0: '#e74c3c', 3.0: '#f39c12', 4.0: '#3498db', 5.0: '#2ecc71'}
    
    for f_val, fractions in fnn_results.items():
        plt.plot(dims, fractions, marker='o', markersize=4, linewidth=2, 
                 color=colors.get(f_val, 'gray'), label=f'Ratio (f) = {f_val}')
        
    # Mark the automatically detected elbow
    plt.axvline(x=m_opt, color='black', linestyle='--', linewidth=2, 
                label=f'Auto-Detected Elbow (m={m_opt})')
    
    plt.title(f'False Nearest Neighbors: Noise Floor Sweep - {tag}', fontsize=14)
    plt.xlabel('Embedding Dimension (m)', fontsize=12)
    plt.ylabel('Fraction of False Neighbors', fontsize=12)
    plt.xticks(dims)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filename = os.path.join(plots_dir, f"{tag}_fnn_sweep.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()


# ── Data Loader ──────────────────────────────────────────────────────

def load_lbnl_train_data(data_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Loads data from the healthy LBNL dataset directly.
    
    Args:
        data_dir: Root directory of the LBNL dataset.
        
    Returns:
        A tuple of (data_array, feature_cols).
    """
    fault_free_path = os.path.join(data_dir, "FCU_FaultFree.csv")
    
    if not os.path.exists(fault_free_path):
        raise FileNotFoundError(f"Cannot find {fault_free_path}. Please check your --data-dir.")

    print(f"\n[1/2] Loading healthy data from {fault_free_path}...", flush=True)
    start_load = time.time()
    
    df = pd.read_csv(fault_free_path)
    
    # Drop metadata, static setpoints, binary indicators, and step actuators
    cols_to_drop = [
        "Datetime", "FCU_CTRL", "FAN_CTRL",                      # Metadata / Categorical
        "RMCLGSPT", "RMHTGSPT",                                  # Static State Setpoints
        "FCU_CVLV", "FCU_CVLV_DM", "FCU_HVLV", "FCU_HVLV_DM",    # Binary/Discrete Step Valves
        "FCU_CLG_GPM", "FCU_HTG_GPM",                            # Resulting Square-wave Flow Rates
        "FCU_DMPR", "FCU_DMPR_DM", "FCU_SPD", "FCU_WAT",         # Actuators and Fan Speed States
        "FCU_OA_CFM", "FCU_DA_CFM",
        "FCU_CLG_EWT", "FCU_HTG_EWT"
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    feature_cols = df.columns.tolist()
    data_array = df.values.astype(np.float64)
    
    print(f"      Data loaded in {time.time() - start_load:.2f} seconds.", flush=True)
    return data_array, feature_cols


# ── TISEAN helpers ────────────────────────────────────────────────────────────

def run_mutual_info(
    sensor_data: np.ndarray, 
    results_dir: str, 
    plots_dir: str, 
    tag: str, 
    max_delay: int
) -> int:
    """
    Run MI on one sensor, return optimal delay d.
    
    Args:
        sensor_data: 1D array of sensor values.
        results_dir: Directory for TISEAN text outputs.
        plots_dir: Directory for visualization plots.
        tag: Unique identifier for the feature.
        max_delay: Maximum delay window to search.
        
    Returns:
        The optimal time delay d_opt.
    """
    mi_in = os.path.join(results_dir, f"{tag}_mi_input.txt")
    mi_out = os.path.join(results_dir, f"{tag}_mi.txt")

    np.savetxt(mi_in, sensor_data, fmt="%.8f")
    result = subprocess.run(
        [MUTUAL_EXE, "-D", str(max_delay), mi_in, "-o", mi_out],
        capture_output=True,
    )
    if os.path.exists(mi_in):
        os.remove(mi_in)
        
    if result.returncode != 0:
        print(f"  WARNING: mutual info failed (exit {result.returncode}) for {tag}, "
              "falling back to d=1", flush=True)
        return 1

    try:
        mutual_data = np.loadtxt(mi_out)
    except Exception as e:
        print(f"  WARNING: Failed to load MI results for {tag}: {e}. Falling back to d=1.")
        return 1

    delays = mutual_data[:, 0].astype(int)
    mi_raw = mutual_data[:, 1]

    # Robust local minimum search with smoothing
    mi_smoothed = smooth_signal(mi_raw, window_size=3)
    
    d_opt = int(delays[-1])  # Default to last if no minimum found
    for i in range(1, len(mi_smoothed) - 1):
        # Check if it's a true local valley
        if mi_smoothed[i] < mi_smoothed[i - 1] and mi_smoothed[i] < mi_smoothed[i + 1]:
            # Ignore high-frequency noise spikes below a 20-minute horizon
            if delays[i] < 20:
                continue

            d_opt = int(delays[i])
            break
            
    # Generate and save plot
    plot_mi_curve(delays, mi_raw, mi_smoothed, d_opt, tag, plots_dir)
    
    return d_opt


def run_fnn(
    sensor_data: np.ndarray, 
    d_opt: int, 
    results_dir: str, 
    plots_dir: str,
    tag: str, 
    max_rows: int,
    max_emb_dim: int = MAX_EMB_DIM,
    no_f_sweep: bool = False,
    fnn_delta: int = FNN_PLATEAU_DELTA
) -> int:
    """
    Run FNN on one sensor with given delay d, return optimal embedding dim m.
    
    Args:
        sensor_data: 1D array of sensor values.
        d_opt: Optimal time delay d.
        results_dir: Directory for TISEAN text outputs.
        tag: Unique identifier for the feature.
        max_rows: Maximum rows to feed to TISEAN.
        
    Returns:
        The optimal embedding dimension m_opt.
    """
    fnn_in = os.path.join(results_dir, f"{tag}_fnn_input.txt")
    np.savetxt(fnn_in, sensor_data[:max_rows], fmt="%.8f")
    
    fnn_results = {}
    dims = None
    
    # ── 1. EXECUTE FNN (Sweep vs. Default) ──
    if no_f_sweep:
        print("  Running single FNN with TISEAN default ratio...", flush=True)
        fnn_out = os.path.join(results_dir, f"{tag}_fnn_default.txt")
        
        # Notice: No "-f" argument in this subprocess array
        result = subprocess.run(
            [FNN_EXE, "-d", str(d_opt), "-M", f"1,{max_emb_dim}", fnn_in, "-o", fnn_out],
            capture_output=True,
        )
        
        if result.returncode == 0 and os.path.exists(fnn_out):
            try:
                data = np.loadtxt(fnn_out)
                if data.ndim == 1: data = data.reshape(1, -1)
                dims = data[:, 0].astype(int)
                fnn_results["Default"] = data[:, 1]
            except Exception as e:
                print(f"  WARNING: Could not parse default output for {tag}: {e}")
    else:
        f_sweep = [1.5, 2.0, 3.0, 4.0, 5.0, 10.0]
        for f_val in f_sweep:
            fnn_out = os.path.join(results_dir, f"{tag}_fnn_f{f_val}.txt")
            
            # Notice: Contains the "-f" argument
            result = subprocess.run(
                [FNN_EXE, "-d", str(d_opt), "-M", f"1,{max_emb_dim}", "-f", str(f_val), fnn_in, "-o", fnn_out],
                capture_output=True,
            )
            
            if result.returncode == 0 and os.path.exists(fnn_out):
                try:
                    data = np.loadtxt(fnn_out)
                    if data.ndim == 1: data = data.reshape(1, -1)
                    if dims is None: dims = data[:, 0].astype(int)
                    fnn_results[f_val] = data[:, 1]
                except Exception as e:
                    print(f"  WARNING: Could not parse f={f_val} output for {tag}: {e}")
                
    if os.path.exists(fnn_in):
        os.remove(fnn_in)

    if not fnn_results:
        print(f"  [ERROR] All FNN executions failed for {tag}. Fallback m=6.")
        return 6

    # ── 2. AUTOMATED ELBOW DETECTION ──
    # If sweeping, use f=2.0 as baseline. If default, just use the only curve we have.
    target_curve = fnn_results.get(3.0, list(fnn_results.values())[-1])
    m_opt = int(dims[-1])
    
    for i in range(1, len(target_curve)):
        delta = target_curve[i-1] - target_curve[i]
        if delta < fnn_delta:
            if i + 1 < len(target_curve):
                next_delta = target_curve[i] - target_curve[i+1]
                if next_delta < fnn_delta:
                    m_opt = int(dims[i-1])
                    print(f"  [AUTO] Invariant elbow locked at m={m_opt}")
                    break

    # ── 3. VISUAL PROOF ──
    plot_fnn_sweep(dims, fnn_results, m_opt, tag, plots_dir)
    
    return m_opt

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Upgraded argument parser to handle new optimization and visualization flags.
    """
    p = argparse.ArgumentParser(description="TISEAN MI + FNN parameter discovery (LBNL Dataset)")
    p.add_argument("--data-dir", required=True, help="Path to the LBNL_FDD_Dataset_FCU folder")
    p.add_argument("--results-dir", default="tisean_results_lbnl",
                   help="Output directory (default: tisean_results_lbnl)")
    p.add_argument("--features", type=str, 
                   help="Comma-separated list of feature names or indices to process (e.g., 'RM_TEMP,FCU_DAT' or '0,4,16')")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                   help=f"Override default row limit (default: {DEFAULT_MAX_ROWS})")
    p.add_argument("--max-delay", type=int, default=DEFAULT_MAX_DELAY,
                   help=f"Override Mutual Information search window (default: {DEFAULT_MAX_DELAY})")
    p.add_argument("--fnn-delta", type=float, default=FNN_PLATEAU_DELTA,
                    help=f"Delta threshold to determine if the FNN plot plateaued (default: {FNN_PLATEAU_DELTA})")
    p.add_argument("--max-emb-dim", type=int, default=10,
                   help="Maximum embedding dimension to search (default: 10)")
    p.add_argument("--no-f-sweep", action="store_true",
                   help="Disable the -f ratio sweep and use TISEAN's native default ratio (f=2.0).")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    plots_mi_dir = os.path.join(results_dir, "plots_MI")
    plots_fnn_dir = os.path.join(results_dir, "plots_FNN")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_mi_dir, exist_ok=True)
    os.makedirs(plots_fnn_dir, exist_ok=True)

    # Verify TISEAN binaries
    for exe in (MUTUAL_EXE, FNN_EXE):
        if not os.path.isfile(exe):
            raise FileNotFoundError(
                f"TISEAN binary not found: {exe}\n"
                "Make sure Tisean_3.0.1/source_c is compiled and in the correct path."
            )

    print(f"=== TISEAN parameter discovery: LBNL DATASET (Refactored) ===", flush=True)
    print(f"Results → {results_dir}", flush=True)

    data, feature_names = load_lbnl_train_data(args.data_dir)
    n_samples, n_features = data.shape
    
    # Handle feature selection
    selected_indices = list(range(n_features))
    if args.features:
        selected_indices = []
        req_features = [f.strip() for f in args.features.split(",")]
        for f in req_features:
            if f.isdigit():
                idx = int(f)
                if 0 <= idx < n_features:
                    selected_indices.append(idx)
                else:
                    print(f"  WARNING: Feature index {idx} out of range (0-{n_features-1}).")
            elif f in feature_names:
                selected_indices.append(feature_names.index(f))
            else:
                print(f"  WARNING: Feature name '{f}' not found in dataset.")

    print(f"\n[2/2] Data processing starting...", flush=True)
    print(f"      Selected: {len(selected_indices)} / {n_features} features")
    print(f"      Configuration: max_rows={args.max_rows:,}, max_delay={args.max_delay}\n", flush=True)

    d_all = np.zeros(n_features, dtype=int)
    m_all = np.zeros(n_features, dtype=int)

    for i in selected_indices:
        name = feature_names[i]
        print(f"{'='*60}", flush=True)
        print(f"Processing Feature {i}: {name}", flush=True)
        tag = f"feat{i}_{name.replace('-', '_').replace(' ', '_')}"

        sensor = data[:, i]

        # ── INPUT SANITIZATION ───────────────────────────────────────────────
        # Check for zero variance (flatline) or NaN values to prevent TISEAN segfaults
        if np.nanstd(sensor) == 0 or np.isnan(sensor).any():
            print(f"  [SAFETY] Feature '{name}' contains zero variance or invalid values.", flush=True)
            print(f"           Applying engineering fallback (d_opt=1, m_opt=1).", flush=True)
            d_all[i] = 1
            m_all[i] = 1
            continue

        # ── MUTUAL INFORMATION ───────────────────────────────────────────────
        print(f"  Running Mutual Information Discovery...", flush=True)
        t0 = time.time()
        d_opt = run_mutual_info(sensor, results_dir, plots_mi_dir, tag, args.max_delay)
        t1 = time.time()
        d_all[i] = d_opt
        print(f"  --> d_optimal = {d_opt}  (took {t1 - t0:.2f}s)", flush=True)

        # ── FALSE NEAREST NEIGHBORS ─────────────────────────────────────────
        print(f"  Running False Nearest Neighbors (d={d_opt})...", flush=True)
        t2 = time.time()
        m_opt = run_fnn(sensor, d_opt, results_dir, plots_fnn_dir, tag, args.max_rows, args.max_emb_dim, args.no_f_sweep, args.fnn_delta)
        t3 = time.time()
        m_all[i] = m_opt
        print(f"  --> m_optimal = {m_opt}  (took {t3 - t2:.2f}s)\n", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("SUMMARY: Optimal Parameters (Processed Features)")
    print(f"{'='*60}")
    print(f"{'ID':<4} {'Feature':<30} {'d (Delay)':>10} {'m (Dim)':>10}")
    print("-" * 60)
    for i in selected_indices:
        print(f"{i:<4} {feature_names[i]:<28} {d_all[i]:>10} {m_all[i]:>10}")
    print()

    # Save results
    # Generate a clean, human-readable structural mapping
    parameter_map = {}
    for i in selected_indices:
        parameter_map[feature_names[i]] = {
            "d_delay": int(d_all[i]),
            "m_dimension": int(m_all[i])
        }

    summary = {
        "dataset": "lbnl",
        "total_continuous_features": len(selected_indices),
        "parameters": parameter_map
    }
    
    np.savez(
        os.path.join(results_dir, "summary.npz"),
        d_optimal=d_all,
        m_optimal=m_all,
    )
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRefactored analysis complete. Results saved to {results_dir}/", flush=True)


if __name__ == "__main__":
    main()
