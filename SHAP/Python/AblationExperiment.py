"""
AblationExperiment.py

Violin plot for baseline + ablations (channels 0..5).
Exports each scenario's error distribution to an Excel file (ablation_results.xlsx).

The script:
1. Loads model from a folder
2. Reads test_indices.txt for the test set
3. Baseline errors vs. ablations
4. Violin plot (lightblue, black edges), median & IQR
5. Exports scenario_errs to Excel so you can plot them in Prism
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

def ablate_channels(X_data, channels_to_ablate):
    """
    Zero out the specified channels (list of int indices).
    """
    X_modified = X_data.copy()
    for ch in channels_to_ablate:
        X_modified[:, :, ch] = 0.0
    return X_modified

def main():
    # =========== 0) Plot Styling ===========
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5

    # =========== 1) Random Seeds ===========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # =========== 2) Config ===========
    channels_to_test = [0,1,2,3,4,5]  # ablate all channels
    model_folder = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\trainedBiConvLSTM_0.80_20250316_1115"
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat"

    features = [
        "Deflection",
        "Smoothed Defl",
        "Derivative",
        "RoV",
        "Slope",
        "R^2"
    ]

    # =========== 3) Load Model ===========
    model = tf.keras.models.load_model(model_folder)
    print(f"Loaded model from '{model_folder}'")

    # =========== 4) Load Data ===========
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']
    Y_mat = data['Y']

    X_all = np.transpose(X_mat, (2,1,0))
    Y_all = Y_mat.flatten()
    N = X_all.shape[0]
    print(f"Loaded dataset with {N} samples.")

    # =========== 5) Load test_indices.txt for test set ===========
    test_indices_path = os.path.join(model_folder, "test_indices.txt")
    if not os.path.exists(test_indices_path):
        raise FileNotFoundError(f"No test_indices.txt found in '{model_folder}'.")

    idx_list = []
    with open(test_indices_path, "r") as f:
        lines = f.readlines()
        start_line = 0
        if lines[0].strip().lower().startswith("test indices"):
            start_line = 1
        for line in lines[start_line:]:
            line=line.strip()
            if line:
                idx_list.append(int(line))

    if len(idx_list) < 2:
        raise ValueError("Not enough test indices for ablation experiment.")

    X_test = X_all[idx_list]
    Y_test = Y_all[idx_list]
    test_count = X_test.shape[0]
    print(f"Test set shape: {X_test.shape} => {test_count} curves from the unseen set.")

    # =========== 6) Baseline Errors ===========
    preds = model.predict(X_test).flatten()
    baseline_errors = np.abs(preds - Y_test)

    scenario_errs = []
    scenario_labels = []

    # baseline
    scenario_errs.append(baseline_errors)
    scenario_labels.append("Baseline")

    # =========== 7) Ablations for each channel ===========
    for ch_idx in channels_to_test:
        X_ablate = ablate_channels(X_test, [ch_idx])
        preds_ablate = model.predict(X_ablate).flatten()
        ablate_errs = np.abs(preds_ablate - Y_test)

        label = f"Ablate {features[ch_idx]}(Ch{ch_idx+1})"
        scenario_errs.append(ablate_errs)
        scenario_labels.append(label)

        print(f"{label}: median={np.median(ablate_errs):.4f}, iqr=({np.percentile(ablate_errs,25):.4f},{np.percentile(ablate_errs,75):.4f})")

    # =========== 8) Violin Plot ===========

    fig, ax = plt.subplots(figsize=(9,4))
    # scenario_errs => list of arrays, each is shape (test_count,)
    parts = ax.violinplot(scenario_errs,
                          positions=range(len(scenario_errs)),
                          showmeans=False, showextrema=False, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # overlay median & IQR lines
    for i, errs in enumerate(scenario_errs):
        q1 = np.percentile(errs, 25)
        median_val = np.median(errs)
        q3 = np.percentile(errs, 75)

        ax.plot([i, i], [q1, q3], color='red', lw=1.5)   # IQR in red
        ax.plot([i-0.2, i+0.2], [median_val, median_val], color='black', lw=2)

    ax.set_xticks(range(len(scenario_labels)))
    ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax.set_title("Ablation Experiment - Violin Plot (Median & IQR)")
    ax.set_ylabel("Absolute Error")
    plt.tight_layout()
    plt.show()

    # =========== 9) Export scenario_errs to Excel ===========
    # We'll make each scenario a column in a DataFrame
    # The length is test_count
    df_dict = {}
    for lbl, arr in zip(scenario_labels, scenario_errs):
        df_dict[lbl] = arr

    df_errs = pd.DataFrame(df_dict)
    excel_outfile = "ablation_results.xlsx"
    df_errs.to_excel(excel_outfile, index=False)
    print(f"Exported ablation scenario errors to '{excel_outfile}'")

    print("Ablation violin plot complete.")

if __name__ == "__main__":
    main()
