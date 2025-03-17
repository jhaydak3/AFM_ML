"""
trainFractionModelClassification.py

Classification version of trainFractionalModel.py:
 - Splits data into training & test sets (train_frac for training).
 - Builds & trains a CNN + biLSTM classification model with 2 output units, 
   final softmax for binary classification: 'reject' vs 'accept'.
 - Evaluates on the unseen test set, prints classification metrics.
 - Saves the model & test indices in a folder named after the fraction & datetime.

Requires a .mat file with:
 - X: [6, time, N]  (features, time, samples) -> We'll transpose to (N, time, 6)
 - Y: [N, 1]        (0 or 1 integer labels for classification)
 - Possibly other fields if needed.

Usage:
  Adjust train_frac, epochs, batch_size, filter_size, etc. 
"""

import os
import datetime
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ReLU, 
                                     MaxPooling1D, Bidirectional, LSTM, 
                                     GlobalAveragePooling1D, Dense, Softmax)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

def build_model_classification(num_features, sequence_length, filter_size=7, l2_reg=1e-4):
    """
    CNN + biLSTM classification model. 
    Final Dense layer has 2 units, followed by a Softmax for binary classification.
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
    
    # Global Average Pooling
    x = GlobalAveragePooling1D(name='globalPool')(x)

    # Fully connected -> ReLU -> final 2-unit Dense -> Softmax
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='fc1')(x)
    x = ReLU(name='relu_fc1')(x)
    x = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='fc_out')(x)
    outputs = Softmax(name='softmax')(x)  # final softmax for 2-class classification

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Configuration
    mat_file_path = r"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"
    train_frac = 0.8     # fraction for training
    epochs     = 30      # or 200 if you want a longer training
    batch_size = 32
    lr         = 1e-4
    l2_reg     = 1e-4
    filter_size= 7

    # Load data
    data = scipy.io.loadmat(mat_file_path)
    X_mat = data['X']  # shape [6, time, N]
    
    # We assume your label field is 'goodOrBad' or something similar. 
    # Adjust if needed:
    Y_mat = data['goodOrBad'] if 'goodOrBad' in data else data['Y']

    # Convert X to (N, time, 6)
    X_all = np.transpose(X_mat, (2, 1, 0))
    N = X_all.shape[0]

    # Convert Y to 0/1 int array
    Y_all = Y_mat.flatten().astype(int)
    print(f"Loaded data with {N} samples from '{mat_file_path}'.")
    print(f"Label distribution: #0={np.sum(Y_all==0)}, #1={np.sum(Y_all==1)}")

    # Shuffle & split
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_idx = int(train_frac * N)
    train_indices = indices[:split_idx]
    test_indices  = indices[split_idx:]

    X_train = X_all[train_indices]
    Y_train = Y_all[train_indices]
    X_test  = X_all[test_indices]
    Y_test  = Y_all[test_indices]
    print(f"Train set size: {X_train.shape[0]} ({train_frac:.2f}), Test set size: {X_test.shape[0]} ({1-train_frac:.2f})")

    # Build model
    num_features   = X_all.shape[-1]
    sequence_length= X_all.shape[1]
    model = build_model_classification(num_features, sequence_length, filter_size, l2_reg)

    # Compile (sparse_categorical_crossentropy since Y is integer 0..1)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train
    history = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1)

    # Evaluate on unseen test set
    loss_test, acc_test = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nTest Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}")

    # Additional metrics (Precision, Recall, F1, etc.)
    preds_test = model.predict(X_test, verbose=0)  # shape [test_size,2]
    y_pred_classes = np.argmax(preds_test, axis=1)

    cm_test = confusion_matrix(Y_test, y_pred_classes)
    print("Test Confusion Matrix:\n", cm_test)
    print("Test Classification Report:\n", 
          classification_report(Y_test, y_pred_classes, target_names=["0","1"], digits=3))

    # Save model and test indices
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fraction_str  = f"{train_frac:.2f}"
    save_folder   = f"trainedBiConvLSTM_Classification_{fraction_str}_{timestamp_str}"

    os.makedirs(save_folder, exist_ok=True)

    # Save the Keras model (SavedModel format)
    model.save(save_folder)
    print(f"Model saved to '{save_folder}/'")

    # Save the test indices to a .txt file
    txt_path = os.path.join(save_folder, "test_indices.txt")
    with open(txt_path, "w") as f:
        f.write("Test Indices:\n")
        for idx in test_indices:
            f.write(f"{idx}\n")
    print(f"Test indices saved to '{txt_path}'")

    # Print summary
    print("\n=== Training Complete (Classification) ===")
    print(f"Final test accuracy: {acc_test:.4f}")
    print(f"Model + test indices saved to '{save_folder}'")

if __name__ == "__main__":
    main()
