import scipy.io
import numpy as np
import pandas as pd
import os
import time
from glob import glob
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

# Parameters
n_points = 2000
folder_path = r'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\AFM_data\Tubules_python_export_only'
num_curves_to_process = 1000
num_repetitions = 100
save_file_name = 'preprocess_times.xlsx'
num_workers = 15

# Build a global file cache and collect valid curves
global_file_cache = {}
valid_curves = []
mat_files = glob(os.path.join(folder_path, '*.mat'))

for file_path in mat_files:
    data = scipy.io.loadmat(file_path, simplify_cells=True)
    global_file_cache[file_path] = data  # Store file data globally
    if 'indentInfoMap_char' not in data:
        print(f"Skipping {file_path}: 'indentInfoMap_char' not found.")
        continue
    indent_info = np.array(data['indentInfoMap_char'])
    rows, cols = np.where(indent_info == 'Accepted')
    for row, col in zip(rows, cols):
        valid_curves.append((file_path, row, col))

if len(valid_curves) == 0:
    raise ValueError("No valid curves found. Check that files contain 'indentInfoMap_char' with 'Accepted' entries.")

# Helper function to normalize values between 0 and 1.
def normalize_zero_to_one(x):
    x = np.asarray(x)
    min_x = np.min(x)
    max_x = np.max(x)
    return np.zeros_like(x) if max_x == min_x else (x - min_x) / (max_x - min_x)

# Process a single curve using the global file cache.
def process_curve(curve_info):
    file_path, row, col = curve_info
    data = global_file_cache[file_path]

    deflection = np.array(data['ExtDefl_Matrix'][row][col])
    extension = np.array(data['Ext_Matrix'][row][col])

    normalized_deflection_orig = normalize_zero_to_one(deflection)
    normalized_extension = normalize_zero_to_one(extension)

    extension_interp = np.linspace(0, 1, n_points)
    interp_func = interp1d(normalized_extension, normalized_deflection_orig, kind='linear', fill_value='extrapolate')
    deflection_interp = interp_func(extension_interp)

    # Smoothing using a moving average via np.convolve.
    smooth_factor = max(1, int(n_points * 0.006))
    smoothed_defl = np.convolve(deflection_interp, np.ones(smooth_factor) / smooth_factor, mode='same')

    # First derivative (as a proxy for the 8th-order derivative).
    eighth_order_derivative = np.gradient(smoothed_defl, edge_order=2)
    normalized_eighth_order_derivative = normalize_zero_to_one(eighth_order_derivative)

    # Rolling variance (RoV) computed via convolution over a window of 35 points.
    window_size = 35
    kernel = np.ones(window_size) / window_size
    mean_rolling = np.convolve(smoothed_defl, kernel, mode='same')
    mean_sq_rolling = np.convolve(smoothed_defl**2, kernel, mode='same')
    rov = mean_sq_rolling - mean_rolling**2
    normalized_rov = normalize_zero_to_one(rov)

    # Linear fit and R-squared computation.
    slope, intercept = np.polyfit(extension_interp, smoothed_defl, 1)
    linear_fit = slope * extension_interp + intercept
    residuals = smoothed_defl - linear_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((smoothed_defl - np.mean(smoothed_defl))**2)
    rsq = 1 - (ss_res / ss_tot)

    normalized_slope = normalize_zero_to_one(slope)
    normalized_rsq = normalize_zero_to_one(rsq)

    features = np.vstack([
        deflection_interp,
        smoothed_defl,
        normalized_eighth_order_derivative,
        normalized_rov,
        np.full(n_points, normalized_slope),
        np.full(n_points, normalized_rsq)
    ])

    return features

# Preprocessing and timing loop (excluding file loading since it's done upfront).
preprocessing_times = []
for iteration in range(num_repetitions):
    print(f"Processing iteration {iteration+1}/{num_repetitions}...")
    start_time = time.time()

    selected_indices = np.random.choice(len(valid_curves), num_curves_to_process, replace=False)
    selected_curves = [valid_curves[idx] for idx in selected_indices]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_curve, selected_curves))

    elapsed_time = time.time() - start_time
    preprocessing_times.append(elapsed_time)
    print(f"Iteration {iteration+1} completed in {elapsed_time:.2f} seconds.")

pd.DataFrame(preprocessing_times, columns=["PreprocessingTime"]).to_excel(save_file_name, index=False)
print(f"Times saved to {save_file_name}")
