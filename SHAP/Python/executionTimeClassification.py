"""
executionTimeClassification.py

Measures:
1) Training time (N1 times) of a 2-class CNN+biLSTM classification model (final softmax).
2) Evaluation time (N2 times), each predicting on 1000 random curves from the dataset.

Saves results in "execution_times_classification.xlsx" with two sheets:
 - "TrainTimes" for the N1 runs
 - "EvalTimes" for the N2 repeated evaluations

Steps:
1. Load classification data from .mat (X with shape [6, time, N], Y with 0/1).
2. Repeatedly train the model N1 times, recording each run's time.
3. After the final run, use that model to measure inference time N2 times.

Adjust hyperparameters (EPOCHS, BATCH_SIZE, etc.) as desired for your scenario.
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
                                     GlobalAveragePooling1D, Dense, Softmax)
from tensorflow.keras.optimizers import Adam

def build_classification_model(num_features, sequence_length, filter_size=7, l2_reg=1e-4):
    """
    CNN + biLSTM classification model for 2 classes, final Softmax.
    """
    inputs = Input(shape=(sequence_length, num_features), name='input')

    # First convolution block
    x = Conv1D(filters=32, kernel_size=filter_size, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool1')(x)

    # Second convolution block
    x = Conv1D(filters=64, kernel_size=filter_size, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool2')(x)

    # BiLSTM
    x = Bidirectional(LSTM(100, return_sequences=True), name='lstmSeq')(x)
    x = GlobalAveragePooling1D(name='globalPool')(x)

    # Dense -> ReLU -> final 2-unit dense -> Softmax
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='fc1')(x)
    x = ReLU(name='relu_fc1')(x)
    x = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='fc_out')(x)
    outputs = Softmax(name='softmax')(x)  # shape (None,2)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # 1) Configuration
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"

    N1 = 10   # number of repeated training runs
    N2 = 1000 # number of repeated evaluation runs

    # Classification hyperparams
    EPOCHS = 30        # or more if you want a full run, but 5 is a decent quick benchmark
    BATCH_SIZE = 32
    INITIAL_LR = 1e-4
    L2_REG = 1e-4
    FILTER_SIZE = 7

    # 2) Load Data
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']          # shape: (6, time, N)
    Y_mat = data['goodOrBad']  # or data['Y'] if that's your label field
    X_all = np.transpose(X_mat, (2,1,0))  # => (N, time, 6)
    Y_all = Y_mat.flatten().astype(int)

    N = X_all.shape[0]
    print(f"Loaded classification data with {N} samples from '{mat_file_path}'.")

    # GPU config
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth set.")
        except RuntimeError as e:
            print("Error setting GPU memory growth:", e)
    else:
        print("No GPU found or not configured.")

    # 3) Repeated Training Time (N1)
    train_times = []
    final_model = None

    for i in range(N1):
        print(f"\n=== Training run {i+1}/{N1} ===")
        # Build classification model
        model = build_classification_model(num_features=6, sequence_length=X_all.shape[1],
                                           filter_size=FILTER_SIZE, l2_reg=L2_REG)
        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        start_time = time.time()
        model.fit(X_all, Y_all, 
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  verbose=1)
        end_time = time.time()

        elapsed_s = end_time - start_time
        train_times.append(elapsed_s)
        print(f"Train run {i+1} took {elapsed_s:.2f} seconds.")

        final_model = model  # store last model

    # 4) Evaluate Time (N2) with final_model
    # We'll do repeated predictions on random 1000-curve subsets
    eval_times = []
    if final_model is None:
        raise RuntimeError("No final model found after training loops.")

    for j in range(N2):
        subset_idx = np.random.choice(N, 1000, replace=False)
        X_subset = X_all[subset_idx]

        start_eval = time.time()
        _ = final_model.predict(X_subset, verbose=0)
        end_eval = time.time()

        e_time = end_eval - start_eval
        eval_times.append(e_time)
        if (j+1) % 50 == 0:
            print(f"Eval run {j+1}/{N2}: {e_time:.4f}s for 1000 samples")

    # 5) Export to "execution_times_classification.xlsx"
    import pandas as pd

    df_train = pd.DataFrame({"TrainTime_s": train_times})
    df_eval  = pd.DataFrame({"EvalTime_s": eval_times})

    excel_outfile = "execution_times_classification.xlsx"
    with pd.ExcelWriter(excel_outfile) as writer:
        df_train.to_excel(writer, sheet_name="TrainTimes", index=False)
        df_eval.to_excel(writer, sheet_name="EvalTimes", index=False)

    # 6) Print summary
    mean_train = np.mean(train_times)
    std_train  = np.std(train_times)
    mean_eval  = np.mean(eval_times)
    std_eval   = np.std(eval_times)

    print("\n=== Classification Execution Time Summary ===")
    print(f"Training runs (N1={N1}, EPOCHS={EPOCHS}) => mean={mean_train:.2f}s, std={std_train:.2f}s")
    print(f"Evaluation runs (N2={N2}, each on 1000 random curves) => mean={mean_eval:.4f}s, std={std_eval:.4f}s")
    print(f"Saved run times to '{excel_outfile}'")

if __name__ == "__main__":
    main()
