import numpy as np
import pandas as pd

def calculate_metrics(test_healthy_scores, test_fault_scores, threshold):
    """
    Calculates TPR, FPR, Precision, and F1-score given model anomaly scores.
    """
    # Healthy test data predictions (True Negatives and False Positives)
    fp = np.sum(test_healthy_scores > threshold)
    tn = np.sum(test_healthy_scores <= threshold)
    
    # Faulty test data predictions (True Positives and False Negatives)
    tp = np.sum(test_fault_scores > threshold)
    fn = np.sum(test_fault_scores <= threshold)
    
    # Calculate Statistical Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1_Score": f1
    }

def format_results_table(results_list):
    """Converts the list of dictionaries into a clean Pandas DataFrame for display."""
    df = pd.DataFrame(results_list)
    # Format the metrics for easier reading
    df['TPR'] = (df['TPR'] * 100).map('{:.2f}%'.format)
    df['FPR'] = (df['FPR'] * 100).map('{:.2f}%'.format)
    df['Precision'] = (df['Precision'] * 100).map('{:.2f}%'.format)
    df['F1_Score'] = df['F1_Score'].map('{:.4f}'.format)
    return df