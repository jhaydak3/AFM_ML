"""
LocalExplanations.py

Now:
1. Loads the final model & test indices from test_indices.txt
2. Uses 700 background, 300 test from those indices
3. Creates the violin plot for global channel-level SHAP
4. Exports the shap_per_channel data to Excel (shap_values_global.xlsx)
5. Also exports the local explanation's channel-level data to a second sheet
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.io
import shap
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

def main():
    # =========== 0) Global Plot Styling ===========
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5

    # =========== 1) Set Random Seed ===========
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # =========== 2) Load the Trained Model ===========
    model_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\trainedBiConvLSTM_0.80_20250316_1115"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'")

    # =========== 3) Load Preprocessed Data ===========
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat"
    data = scipy.io.loadmat(mat_file_path)

    X_mat = data['X']  # shape: (6, 2000, N)
    Y_mat = data['Y']  # shape: (N, 1)

    # For unnormalizing (optional)
    maxExtValues = data['maxExtValues'].flatten() if 'maxExtValues' in data else None
    minExtValues = data['minExtValues'].flatten() if 'minExtValues' in data else None

    # Reshape X => (N,2000,6)
    X_all = np.transpose(X_mat, (2, 1, 0))
    Y_all = Y_mat.flatten()
    N = X_all.shape[0]
    print(f"Loaded dataset with {N} samples from '{mat_file_path}'.")

    # =========== 4) Load test_indices from test_indices.txt, pick 700 for bg, 300 for test ===========
    test_indices_path = os.path.join(model_path, "test_indices.txt")
    if not os.path.exists(test_indices_path):
        raise FileNotFoundError(f"No test_indices.txt found in '{model_path}'.")

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

    if len(idx_list) < 1000:
        raise ValueError("Need at least 1000 test indices to pick 700 for background and 300 for test.")

    bg_indices = idx_list[:700]
    test_indices = idx_list[700:1000]

    X_bg = X_all[bg_indices]
    X_test = X_all[test_indices]
    print(f"Background shape: {X_bg.shape} (700 samples)")
    print(f"Test subset shape: {X_test.shape} (300 samples)")

    # =========== 5) SHAP GradientExplainer ===========
    explainer = shap.GradientExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]  # => (300, 2000, 6)

    print(f"SHAP values shape: {shap_values.shape}")

    # =========== 6) Aggregated Channel-Level Feature Importance (Global) ===========
    shap_abs = np.abs(shap_values)             # (300, 2000, 6)
    shap_per_channel = np.mean(shap_abs, axis=1)  # => (300,6)

    features = [
        "Deflection",
        "Smoothed Deflection",
        "Derivative",
        "RoV",
        "Slope",
        "R^2"
    ]

    # ========== PLOT: Violin or Aggregation over shap_per_channel ==========

    fig, ax = plt.subplots(figsize=(8,4))

    data_list = [shap_per_channel[:, i] for i in range(6)]
    parts = ax.violinplot(data_list, positions=range(6),
                          showmeans=False, showextrema=False, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # Add median + IQR in red + black dash
    for i, arr in enumerate(data_list):
        q1 = np.percentile(arr, 25)
        median_val = np.median(arr)
        q3 = np.percentile(arr, 75)
        ax.plot([i, i], [q1, q3], color='red', lw=1.5)     # IQR in red
        ax.plot([i-0.2, i+0.2], [median_val, median_val], color='black', lw=2)

    ax.set_xticks(range(6))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_title("Global Channel-Level Feature Importance (Violin Plot)")
    ax.set_ylabel("Avg |SHAP Value| Over Time")
    plt.tight_layout()
    plt.show()

    # =========== Export shap_per_channel to Excel ===========
    # shape => (300,6). We'll label columns with 'features'.
    df_shap_global = pd.DataFrame(shap_per_channel, columns=features)
    excel_outfile = "shap_values_global.xlsx"
    with pd.ExcelWriter(excel_outfile) as writer:
        df_shap_global.to_excel(writer, sheet_name="Global_SHAP", index=False)
        print(f"Exported global SHAP data to '{excel_outfile}'")

    # =========== 7) Local Explanation for a Single Test Sample ===========
    sample_idx = 0
    local_input = X_test[sample_idx:sample_idx+1]  # shape (1,2000,6)
    local_shap = explainer.shap_values(local_input)
    if isinstance(local_shap, list):
        local_shap = local_shap[0]
    if local_shap.ndim == 4:
        local_shap = local_shap[..., 0]  # => shape (1,2000,6)

    local_shap_abs = np.abs(local_shap[0])   # => (2000,6)
    local_channel_importance = np.mean(local_shap_abs, axis=0)  # => (6,)

    plt.figure(figsize=(6,4))
    plt.bar(features, local_channel_importance, color='skyblue', edgecolor='black')
    plt.title(f"Local Channel-Level Importance (Sample idx={sample_idx})")
    plt.ylabel("Mean |SHAP Value| Over Time")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # =========== Export local explanation data to second sheet ===========
    # shape => (6,). We'll store channel names in rows, local SHAP in single column
    df_local = pd.DataFrame({
        "Channel": features,
        "Local_SHAP": local_channel_importance
    })

    with pd.ExcelWriter(excel_outfile, mode="a", if_sheet_exists="replace") as writer:
        df_local.to_excel(writer, sheet_name="Local_SHAP", index=False)
        print(f"Appended local explanation data to '{excel_outfile}' on sheet 'Local_SHAP'")

    print(f"Local explanation for test sample idx={sample_idx} done.")
    print("SHAP local analysis complete.")

if __name__ == "__main__":
    main()
