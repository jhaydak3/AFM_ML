clear
clc
close all
%% General parameters

helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\helperFunctions";
addpath(helperFunctionsFolder);

%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v4_CNI_predict\training\processed_data_for_classification_CNI_and_PKD_Expanded_classification.mat";
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v4_CNI_predict\training\processed_data_for_classification_tubules_classification.mat";
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\classification_processed_files\processed_features_for_classification_All.mat";
nFeatures = 6;
sequenceLength = 2000;
%% Run the models

% one CNN, stacked biLSTM, ReLu
layers = CNN_stacked_bilstms_relu_classification(nFeatures, sequenceLength, 7);
saveName = "one_CNN_stacked_biLSTM_ReLu_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)

% one CNN, single biLSTM (single output), ReLu
layers = CNN_biLSTM_classification(nFeatures, sequenceLength, 7);
saveName = "one_CNN_single_biLSTMsingleoutput_ReLu_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)



% One CNN, LSTM (sequence), GAP, ReLu
layers = CNN_custom_pooling_after_lstm_relu_classification(nFeatures, sequenceLength, 7);
saveName = "one_CNN_LSTMsequence_GAP_ReLu_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)

% Two CNN, LSTM (sequence), GAP, no ReLu
layers = CNN_custom_pooling_after_lstm_2conv_classification(nFeatures, sequenceLength, 7);
saveName = "two_CNN_LSTMsequence_GAP_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)

% Two CNN, LSTM (sequence), GAP, ReLu
layers = CNN_custom_pooling_after_lstm_2conv_relu_classification(nFeatures, sequenceLength, 7);
saveName = "two_CNN_LSTMsequence_GAP_ReLu_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)

% Three CNN, LSTM (sequence), GAP, ReLu
layers = CNN_custom_pooling_after_lstm_3conv_relu_classification(nFeatures, sequenceLength, 9);
saveName = "three_CNN_LSTMsequence_GAP_ReLu_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)



% Sotres (2022) ResNet50-1D
layers = CNN_Sotres_ResNet50_classification(1,sequenceLength, 3, 3, 2);
saveName = "Sotres_ResNet50-1D_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, false)

% Sotres (2022) ResNet50-1D, Expanded Features
layers = CNN_Sotres_ResNet50_classification(nFeatures,sequenceLength, 3, 3, 2);
saveName = "Sotres_ResNet50-1D_expanded_features_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)

% Sotres (2022) Original
layers = CNN_Sotres_classification(1,sequenceLength);
saveName = "Sotres_CNN-1D_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, false)

% Sotres (2022) Original, expanded features
layers = CNN_Sotres_classification(nFeatures,sequenceLength);
saveName = "Sotres_CNN-1D_expanded_features_classification.mat";
evaluate_model_classification_CNN_custom(layers,preprocessedDataFile, saveName, true)




