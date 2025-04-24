import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Bidirectional, LSTM, GlobalAveragePooling1D, Dense

def build_model(numFeatures, sequenceLength, filterSize):
    # 1. Input layer
    inputs = Input(shape=(sequenceLength, numFeatures), name='input')
    
    # 2. First convolutional block
    x = Conv1D(filters=32, kernel_size=filterSize, padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool1')(x)
    
    # 3. Second convolutional block
    x = Conv1D(filters=64, kernel_size=filterSize, padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool2')(x)
    
    # 4. Bidirectional LSTM with output sequences
    x = Bidirectional(LSTM(100, return_sequences=True), name='lstmSeq')(x)
    
    # 5. Global average pooling (1D)
    x = GlobalAveragePooling1D(name='globalPool')(x)
    
    # 6. Fully connected layers
    x = Dense(128, name='fc1')(x)
    x = ReLU(name='relu_fc1')(x)
    outputs = Dense(1, name='fc_out')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
numFeatures = 6
sequenceLength = 2000
filterSize = 7  # Adjust as needed

model = build_model(numFeatures, sequenceLength, filterSize)
model.summary()
