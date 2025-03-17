"""
shapAnalysisClassification.py

For a binary classification model with final shape [None,2] and a softmax.
We interpret class=1 SHAP values, compute global channel-level importance
by averaging absolute SHAP values over time for each sample, and create a violin
plot with overlaid scatter points (showing all 300 samples per channel). We then
export the per-sample (300 x 6) values to Excel and provide a local explanation
for sample_idx=0.

Auto-fix: If SHAP returns a list (one per test sample), we stack them.
If the resulting array has an extra last dimension, we remove it.
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.io
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

def reorder_shap_dimensions(shap_values, desired_batch_size=300, desired_time=2000, desired_features=6):
    """
    Try to reorder a 3D array to (batch, time, features) by checking dimensions.
    If no match is found, return the array as-is.
    """
    if shap_values.shape == (desired_batch_size, desired_time, desired_features):
        return shap_values
    dims = shap_values.shape
    # Look for dims that match (approximately) desired sizes:
    # We assume the largest dim is time (2000), then next is batch (300), then features (6)
    if len(dims) == 3:
        # A simple heuristic: sort dims descending.
        sorted_dims = sorted(dims, reverse=True)
        # Typically, time=2000 (largest), batch=300, features=6
        # If dims are e.g. (2000, 6, 300), then we need to transpose to (300,2000,6)
        if dims[0] == sorted_dims[0] and dims[1] == sorted_dims[2] and dims[2] == sorted_dims[1]:
            return np.transpose(shap_values, (2,0,1))
    return shap_values

def main():
    # ---------- 0) Global Plot Styling ----------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5

    # ---------- 1) Set Random Seed ----------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ---------- 2) Load the Trained Classification Model ----------
    model_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\SHAP\Python\trainedBiConvLSTM_Classification_0.80_20250316_1927"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded classification model from '{model_path}'")
    print("Model output shape:", model.output_shape)

    # ---------- 3) Load Preprocessed Classification Data ----------
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']  # expected shape: (6, time, N)
    N = X_mat.shape[-1]
    X_all = np.transpose(X_mat, (2, 1, 0))  # => (N, time, 6)
    print(f"Loaded dataset with {N} samples from '{mat_file_path}' (classification).")
    
    # ---------- 4) Load test_indices from file; pick 700 for background & 300 for test ----------
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
        raise ValueError("Need >=1000 test indices to pick 700 for background + 300 for test.")

    bg_indices = idx_list[:700]
    test_indices = idx_list[700:1000]

    X_bg = X_all[bg_indices]
    X_test = X_all[test_indices]
    print(f"Background shape: {X_bg.shape} -> (700, time, 6)")
    print(f"Test set shape: {X_test.shape} -> (300, time, 6)")

    # ---------- 5) SHAP Explainer for Class=1 ----------
    explainer = shap.GradientExplainer(model, X_bg)
    shap_values_list = explainer.shap_values(X_test)
    print("Length of shap_values_list:", len(shap_values_list))
    for i in range(min(3, len(shap_values_list))):
        print(f"Shape of shap_values_list[{i}]:", shap_values_list[i].shape)
        
    # Here, since the model output is (None,2) but SHAP returns a list of length 300,
    # we assume each element is the SHAP for one sample.
    # Therefore, stack the list along a new axis.
    if len(shap_values_list) > 1:
        shap_values = np.stack(shap_values_list, axis=0)
        print("Stacked shap_values_list into array with shape:", shap_values.shape)
    else:
        shap_values = shap_values_list[0]

    # If shap_values is 4D, remove trailing dim.
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]
        print("Reduced 4D array to 3D:", shap_values.shape)

    # Now we expect shap_values to have shape (300, time, features) but we see:
    # If not, try to reorder dimensions.
    if shap_values.shape[0] != X_test.shape[0]:
        print("Attempting to reorder dimensions of shap_values...")
        shap_values = np.transpose(shap_values, (2,0,1))
    # Check final shape:
    print(f"Final shap_values shape: {shap_values.shape}")
    # Ideally, it should be (300, 2000, 6)

    # ---------- 6) Global SHAP: Compute per-sample channel-level values ----------
    # We compute absolute SHAP and average over time to yield a (300,6) array.
    shap_abs = np.abs(shap_values)  # (300, time, 6)
    shap_per_channel = np.mean(shap_abs, axis=1)  # (300,6)
    print("shap_per_channel shape:", shap_per_channel.shape)

    features = [
        "Deflection",
        "Smoothed Deflection",
        "Derivative",
        "RoV",
        "Slope",
        "R^2"
    ]
    channels_count = shap_per_channel.shape[1]
    if channels_count != 6:
        print(f"Warning: Expected 6 channels, got {channels_count}.")
        features = features[:channels_count]

    # ---------- 7) Plot Global SHAP as Violin Plot with Scatter ----------
    data_list = [shap_per_channel[:, i] for i in range(channels_count)]
    fig, ax = plt.subplots(figsize=(8,4))
    parts = ax.violinplot(data_list, positions=range(channels_count),
                          showmeans=False, showextrema=False, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # Overlay scatter points, median and IQR lines
    for i, arr in enumerate(data_list):
        x_jitter = np.random.normal(i, 0.06, size=len(arr))
        ax.scatter(x_jitter, arr, color='blue', alpha=0.4, s=15)
        q1 = np.percentile(arr, 25)
        median_val = np.median(arr)
        q3 = np.percentile(arr, 75)
        ax.plot([i, i], [q1, q3], color='red', lw=1.5)   # IQR in red
        ax.plot([i-0.2, i+0.2], [median_val, median_val], color='black', lw=2)
    ax.set_xticks(range(channels_count))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_title("Global Channel-Level SHAP (Class=1) with Scatter")
    ax.set_ylabel("Avg |SHAP Value| Over Time")
    plt.tight_layout()
    plt.show()

    # ---------- 8) Export Global SHAP Data to Excel ----------
    excel_outfile = "shap_values_global_classification.xlsx"
    df_shap_global = pd.DataFrame(shap_per_channel, columns=features)
    with pd.ExcelWriter(excel_outfile) as writer:
        df_shap_global.to_excel(writer, sheet_name="Global_SHAP", index=False)
        print(f"Exported global SHAP data ({df_shap_global.shape[0]}x{df_shap_global.shape[1]}) to '{excel_outfile}'")

    # ---------- 9) Local Explanation for sample_idx=0 ----------
    sample_idx = 0
    local_input = X_test[sample_idx:sample_idx+1]  # shape (1, time, 6)
    local_shap_list = explainer.shap_values(local_input)
    if len(local_shap_list) == 2:
        local_shap_class = local_shap_list[1]
    else:
        local_shap_class = local_shap_list[0]

    if local_shap_class.ndim == 4:
        local_shap_class = local_shap_class[..., 0]
    # Attempt to reorder if needed:
    if local_shap_class.shape[0] != 1:
        local_shap_class = np.transpose(local_shap_class, (2,0,1))
        print("Transposed local_shap to fix shape.")
    print("Local shap shape:", local_shap_class.shape)  # expect (1, time, features)
    local_shap_abs = np.abs(local_shap_class[0])
    if local_shap_abs.shape[1] == channels_count:
        local_channel_importance = np.mean(local_shap_abs, axis=0)
    else:
        local_channel_importance = np.mean(local_shap_abs.reshape(local_shap_abs.shape[0], -1), axis=0)
    print("Local channel importance:", local_channel_importance)

    plt.figure(figsize=(6,4))
    plt.bar(features, local_channel_importance, color='skyblue', edgecolor='black')
    plt.title(f"Local SHAP (Class=1, Sample idx={sample_idx})")
    plt.ylabel("Mean |SHAP Value| Over Time")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ---------- 10) Export Local Explanation to Excel (Second Sheet) ----------
    df_local = pd.DataFrame({
        "Channel": features,
        "Local_SHAP": local_channel_importance
    })
    with pd.ExcelWriter(excel_outfile, mode="a", if_sheet_exists="replace") as writer:
        df_local.to_excel(writer, sheet_name="Local_SHAP", index=False)
        print(f"Appended local explanation data to '{excel_outfile}' on sheet 'Local_SHAP'")

    print(f"Local explanation for sample idx={sample_idx} done.")
    print("Classification SHAP local analysis complete.")

if __name__ == "__main__":
    main()
