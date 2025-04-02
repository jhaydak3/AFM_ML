%%

clear
clc
close all
%% General parameters
sequenceLength = 2000;
nFeatures = 6;
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_podocytes.mat";
% Path to helper functions
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder)


%% Run the models

% CNN 2 Conv block, LSTM sequence mode, GAP, dense
layers = CNN_custom_pooling_after_lstm_2conv_relu_classification(nFeatures, sequenceLength, 7);
saveName = "trainedClassificationModels\pooling_after_bilstm_2conv_relu_classification.mat";
train_CNN_model_classification_function(layers,preprocessedDataFile, saveName, true)



