"""
ablationExperimentClassificationAUC.py

For a binary classification model, this script:
  1. Loads the classification model and preprocessed data.
  2. Reads test_indices.txt and uses the test portion (indices 700:1000, i.e. all 300 test curves).
  3. For the baseline and for each channel ablation (channels 0–5), computes the predicted
     probability for class=1 and then calculates the ROC AUC on the entire test set.
  4. Plots a bar chart of the AUC for each scenario.
  5. Exports the AUC results to an Excel file named "ablation_results_auc_classification.xlsx".
  
No extra background sample is needed here—we use all test curves.
  
Requires:
  - sklearn for roc_auc_score.
  - pandas (with openpyxl or xlsxwriter) for Excel export.
  
Adjust file paths as needed.
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def ablate_channels(X_data, channels_to_ablate):
    """
    Returns a copy of X_data with the specified channels zeroed out.
    channels_to_ablate: list of 0-based channel indices.
    """
    X_modified = X_data.copy()
    for ch in channels_to_ablate:
        X_modified[:, :, ch] = 0.0
    return X_modified

def get_auc(model, X_input, Y_true):
    """
    Predicts using the model on X_input, extracts the probability for class 1,
    and computes the ROC AUC using the true labels.
    """
    preds = model.predict(X_input, verbose=0)  # shape: (num_samples, 2)
    prob_class1 = preds[:, 1]  # probability for class 1
    return roc_auc_score(Y_true, prob_class1)

def main():
    # ---------- Global Plot Styling ----------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5

    # ---------- Set Random Seed ----------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ---------- Load the Trained Classification Model ----------
    model_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\trainedBiConvLSTM_Classification_0.80_20250316_1927"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded classification model from '{model_path}'")
    print("Model output shape:", model.output_shape)

    # ---------- Load Preprocessed Classification Data ----------
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']  # Expected shape: (6, time, N)
    # Use 'goodOrBad' if available, else 'Y'
    Y_mat = data['goodOrBad'] if 'goodOrBad' in data else data['Y']
    # Reshape X => (N, time, 6)
    X_all = np.transpose(X_mat, (2, 1, 0))
    N_total = X_all.shape[0]
    print(f"Loaded dataset with {N_total} samples from '{mat_file_path}' (classification).")

    # ---------- Load test_indices and select the test set ----------
    test_indices_path = os.path.join(model_path, "test_indices.txt")
    if not os.path.exists(test_indices_path):
        raise FileNotFoundError(f"No test_indices.txt found in '{model_path}'.")
    idx_list = []
    with open(test_indices_path, "r") as f:
        lines = f.readlines()
        start_line = 0
        if lines and lines[0].strip().lower().startswith("test indices"):
            start_line = 1
        for line in lines[start_line:]:
            line = line.strip()
            if line:
                idx_list.append(int(line))
    if len(idx_list) < 1000:
        raise ValueError("Need >=1000 test indices to pick 700 for background and 300 for test.")
    # For ablation experiments, we use all test curves (indices 700 to 1000, i.e., 300 samples)
    test_indices = idx_list[700:1000]
    X_test = X_all[test_indices]
    print(f"Test set shape: {X_test.shape} -> (300, time, 6)")

    # ---------- Load true labels for the test set ----------
    Y_all = Y_mat.flatten().astype(int)
    Y_test = Y_all[test_indices]

    # ---------- Baseline AUC ----------
    baseline_auc = get_auc(model, X_test, Y_test)
    print(f"Baseline AUC: {baseline_auc:.4f}")

    # ---------- Ablation Experiment: Evaluate AUC for each channel ablation ----------
    features = ["Deflection", "Smoothed Defl", "Derivative", "RoV", "Slope", "R^2"]
    scenario_auc = [baseline_auc]  # first scenario: baseline
    scenario_labels = ["Baseline"]

    for ch in range(6):
        X_test_ablate = ablate_channels(X_test, [ch])
        auc_val = get_auc(model, X_test_ablate, Y_test)
        label = f"Ablate {features[ch]} (Ch{ch+1})"
        scenario_auc.append(auc_val)
        scenario_labels.append(label)
        print(f"{label}: AUC = {auc_val:.4f}")

    # ---------- Plot AUC for each scenario as a Bar Chart ----------
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(scenario_labels, scenario_auc, color='skyblue', edgecolor='black')
    ax.set_ylabel("AUC")
    ax.set_title("Ablation Experiment (Classification): AUC per Scenario")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ---------- Export AUC results to Excel ----------
    df_auc = pd.DataFrame({
        "Scenario": scenario_labels,
        "AUC": scenario_auc
    })
    excel_outfile = "ablation_results_auc_classification.xlsx"
    df_auc.to_excel(excel_outfile, index=False)
    print(f"Exported AUC results to '{excel_outfile}'")

if __name__ == "__main__":
    main()
