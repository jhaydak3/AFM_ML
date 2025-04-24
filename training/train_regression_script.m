%%

clear
clc
close all
%% General parameters
sequenceLength = 2000;
nFeatures = 6;
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All_5000.mat"; 
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_podocytes.mat";
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_Jon_annotated.mat";
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_Evren_annotated.mat"
% Path to helper functions
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder)


%% Run the models

% CNN 2 Conv block, LSTM sequence mode, relu, no augmentation
layers = CNN_custom_pooling_after_bilstm_2conv_relu(nFeatures, sequenceLength, 7);
saveName = "trainedRegressionModels\pooling_after_bilstm_2conv_relu_Evren_Annotated.mat";
train_CNN_model_regression_function(layers,preprocessedDataFile, saveName, true, false)



