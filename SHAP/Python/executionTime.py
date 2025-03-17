"""
executionTime.py

Updated:
1) We train the model N1 times, with verbose=1 so we see training progress.
2) We do NOT retrain for the final evaluation. Instead we use the last trained 
   model from the N1-th run for all inference tests. 
3) Exports train times and eval times to execution_times.xlsx.

Training parameters (from your prior scripts):
 - filterSize=7
 - EPOCHS=200
 - BATCH_SIZE=32
 - LR=1e-4
 - L2=1e-4

N1=10 for repeated training time measurements, 
N2=1000 for repeated evaluation tests on random 1000-curve subsets.

"""

import os
import time
import random
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ReLU, 
                                     MaxPooling1D, Bidirectional, LSTM, 
                                     GlobalAveragePooling1D, Dense)
from tensorflow.keras.optimizers import Adam

# -------- MODEL BUILD --------
def build_model(numFeatures, sequenceLength, filterSize, l2_reg):
    inputs = Input(shape=(sequenceLength, numFeatures), name='input')

    x = Conv1D(filters=32, kernel_size=filterSize, padding='same', name='conv1',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool1')(x)

    x = Conv1D(filters=64, kernel_size=filterSize, padding='same', name='conv2',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool2')(x)

    x = Bidirectional(LSTM(100, return_sequences=True), name='lstmSeq')(x)
    x = GlobalAveragePooling1D(name='globalPool')(x)

    x = Dense(128, name='fc1', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = ReLU(name='relu_fc1')(x)
    outputs = Dense(1, name='fc_out', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # 1) Configuration
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat"
    
    # Run times
    N1 = 10   # training repeats
    N2 = 1000 # evaluation repeats

    # Training hyperparams
    EPOCHS = 200
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4
    L2_REG = 1e-4
    filterSize = 7

    # 2) Load Data
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']  # shape (6, 2000, N)
    Y_mat = data['Y']  # shape (N, 1)

    X_all = np.transpose(X_mat, (2,1,0))
    Y_all = Y_mat.flatten()

    N = X_all.shape[0]
    numFeatures = 6
    sequenceLength = 2000

    print(f"Loaded dataset with {N} samples from '{mat_file_path}'.")

    # GPU config
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth set.")
        except RuntimeError as e:
            print("Error setting memory growth:", e)
    else:
        print("No GPU found or no GPU config set.")

    # 3) Training Time (N1 times), verbose=1
    train_times = []
    final_model = None
    for i in range(N1):
        print(f"\n=== Training run {i+1}/{N1} ===")
        model = build_model(numFeatures, sequenceLength, filterSize, L2_REG)
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                               tf.keras.metrics.MeanSquaredError(name="MSE")])

        start_t = time.time()
        model.fit(X_all, Y_all,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  verbose=1)  # show training progress
        end_t = time.time()

        elapsed_s = end_t - start_t
        train_times.append(elapsed_s)
        print(f"Train run {i+1} took {elapsed_s/60:.2f} min ({elapsed_s:.2f} s).")

        # We'll keep the last trained model for final evaluation
        final_model = model

    # final_model is now the last trained from the N1-th run

    # 4) Evaluate Time (N2 times) with final_model
    #    For each iteration, pick 1000 random curves => measure predict time
    eval_times = []
    for j in range(N2):
        subset_idx = np.random.choice(N, 1000, replace=False)
        X_subset = X_all[subset_idx]

        st_eval = time.time()
        _ = final_model.predict(X_subset, verbose=0)
        et_eval = time.time()

        e_time = et_eval - st_eval
        eval_times.append(e_time)

        if (j+1) % 50 == 0:
            print(f"Evaluation run {j+1}/{N2}, time={e_time:.4f} s for 1000 curves")

    # 5) Export to Excel
    df_train = pd.DataFrame({"TrainTime_s": train_times})
    df_eval  = pd.DataFrame({"EvalTime_s": eval_times})

    excel_outfile = "execution_times.xlsx"
    with pd.ExcelWriter(excel_outfile) as writer:
        df_train.to_excel(writer, sheet_name="TrainTimes", index=False)
        df_eval.to_excel(writer, sheet_name="EvalTimes", index=False)

    # 6) Print summary
    mean_train = np.mean(train_times)
    std_train  = np.std(train_times)
    mean_eval  = np.mean(eval_times)
    std_eval   = np.std(eval_times)

    print("\n=== Execution Time Summary ===")
    print(f"Training runs (N1={N1}, EPOCHS={EPOCHS}) => mean={mean_train:.2f}s, std={std_train:.2f}s")
    print(f"Evaluation runs (N2={N2}, each 1000 curves) => mean={mean_eval:.4f}s, std={std_eval:.4f}s")
    print(f"Saved run times to '{excel_outfile}' (TrainTimes & EvalTimes). Done.")

if __name__ == "__main__":
    main()
