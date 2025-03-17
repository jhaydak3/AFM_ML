"""
trainFractionalModel.py

Trains a BiConvLSTM model on a fraction (e.g., 80%) of the data,
leaving the other fraction (20%) as a test set. Prints final MSE, MAE
in both normalized scale and nm, then saves:
  - The trained model in a folder named after the fraction and datetime
  - A .txt file with the indices of the test set

Usage:
Adjust:
- train_frac = 0.8 (the fraction for training)
- Filter paths as needed
"""

import os
import numpy as np
import datetime
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ReLU, 
                                     MaxPooling1D, Bidirectional, LSTM, 
                                     GlobalAveragePooling1D, Dense)
from tensorflow.keras.optimizers import Adam

# Optional GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs found and memory growth set.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)
else:
    print("No GPU found. Check your CUDA/cuDNN installation if you expect to use a GPU.")

def build_model(numFeatures, sequenceLength, filterSize, l2_reg):
    """
    Builds a CNN + BiLSTM model with L2 regularization on conv and dense layers,
    matching the architecture described in your MATLAB code.
    """
    inputs = Input(shape=(sequenceLength, numFeatures), name='input')
    
    # First convolution block
    x = Conv1D(filters=32, kernel_size=filterSize, padding='same', name='conv1',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool1')(x)
    
    # Second convolution block
    x = Conv1D(filters=64, kernel_size=filterSize, padding='same', name='conv2',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool2')(x)
    
    # BiLSTM (OutputMode="sequence")
    x = Bidirectional(LSTM(100, return_sequences=True), name='lstmSeq')(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D(name='globalPool')(x)
    
    # Fully Connected + ReLU + Output
    x = Dense(128, name='fc1', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = ReLU(name='relu_fc1')(x)
    outputs = Dense(1, name='fc_out', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # ============== Configuration ================
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat"
    
    train_frac = 0.8  # fraction of data to use for training
    numFeatures = 6
    sequenceLength = 2000
    filterSize = 7
    l2_reg = 1e-4
    
    # Training settings
    NUM_EPOCHS = 200
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4
    
    # ============== Load Data from .mat file ================
    data = scipy.io.loadmat(mat_file_path)
    X = data['X']            # shape: (6, 2000, N)
    Y = data['Y']            # shape: (N, 1)
    if 'maxExtValues' not in data or 'minExtValues' not in data:
        raise KeyError("maxExtValues or minExtValues not found in .mat file. Please check your data keys.")
    maxExtValues = data['maxExtValues'].flatten()  # shape (N,)
    minExtValues = data['minExtValues'].flatten()  # shape (N,)
    
    # Reshape X to (N, 2000, 6)
    X = np.transpose(X, (2, 1, 0))  # => (N, 2000, 6)
    Y = Y.flatten()                 # => (N,)
    
    N = X.shape[0]
    print(f"Loaded {N} total samples from {mat_file_path}.")
    
    # ============== Shuffle & Split ================
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_idx = int(N * train_frac)
    
    train_indices = indices[:split_idx]
    test_indices  = indices[split_idx:]
    
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test  = X[test_indices]
    Y_test  = Y[test_indices]
    
    print(f"Training set size: {X_train.shape[0]} (fraction={train_frac})")
    print(f"Test set size:     {X_test.shape[0]} (fraction={1-train_frac})")
    
    # ============== Build and Compile the Model ================
    model = build_model(numFeatures, sequenceLength, filterSize, l2_reg)
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.MeanAbsoluteError(), 
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                           tf.keras.metrics.MeanSquaredError(name="MSE")])
    model.summary()
    
    # ============== Train the Model ================
    print(f"Training on {X_train.shape[0]} curves, validating on {X_test.shape[0]} unseen curves.")
    history = model.fit(X_train, Y_train,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        verbose=1)
    
    # ============== Evaluate on the Unseen Curves ================
    print("Evaluating model on the unseen test set...")
    loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Final (normalized) - Loss(MAE): {loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}")
    
    # ============== Compute Unnormalized Errors in nm ================
    preds_norm = model.predict(X_test).flatten()  # shape (N_test,)
    pred_nm = preds_norm * (maxExtValues[test_indices] - minExtValues[test_indices]) + minExtValues[test_indices]
    true_nm = Y_test * (maxExtValues[test_indices] - minExtValues[test_indices]) + minExtValues[test_indices]
    
    errors_nm = pred_nm - true_nm
    mse_nm = np.mean(errors_nm**2)
    mae_nm = np.mean(np.abs(errors_nm))
    
    print(f"Final (unnormalized, nm) - MSE: {mse_nm:.6f} nm^2, MAE: {mae_nm:.6f} nm")
    
    # ============== Save the Model & Indices ================
    # Create a folder name with fraction + date/time
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fraction_str  = f"{train_frac:.2f}"
    save_folder   = f"trainedBiConvLSTM_{fraction_str}_{timestamp_str}"
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the model in that folder
    model.save(save_folder)
    
    # Save the test indices in a .txt file
    txt_path = os.path.join(save_folder, "test_indices.txt")
    with open(txt_path, "w") as f:
        f.write("Test Indices:\n")
        for idx in test_indices:
            f.write(f"{idx}\n")
    
    print(f"Model saved to '{save_folder}/'")
    print(f"Test indices saved to '{txt_path}'")
    
    # ============== Print Summary ================
    print("\n=== Training Complete ===")
    print(f"Test MSE (normalized): {mse:.6f}")
    print(f"Test MAE (normalized): {mae:.6f}")
    print(f"Test MSE (nm): {mse_nm:.6f} nm^2")
    print(f"Test MAE (nm): {mae_nm:.6f} nm")

if __name__ == "__main__":
    import datetime
    main()
