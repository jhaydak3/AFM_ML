%%

clear
clc
close all
%% General parameters
sequenceLength = 2000;
nFeatures = 6;
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat"; 
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_podocytes.mat";
% Path to helper functions
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder)


%% Run the models

% Current model
% layers = CNN_custom2(nFeatures, sequenceLength, 7);
% saveName = "one_CNN_one_biLSTM.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)

% % Current model
% layers = CNN_custom2(nFeatures, sequenceLength, 7);
% saveName = "one_CNN_one_biLSTM_no_augmentation.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, false)

% Sotres original
layers =  CNN_Sotres(1, sequenceLength);
saveName = "Sotres2022_original_no_augmentation.mat";
general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, false, false)

% % Sotres original, expanded features, no augmentation
% layers =  CNN_Sotres(6, sequenceLength);
% saveName = "Sotres2022_original_expanded_features_no_augmentation.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, false)

% % Sotres original, augmentation
% layers =  CNN_Sotres(1, sequenceLength);
% saveName = "Sotres2022_original.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, false, true)

% % Sotres ResNet50
% layers = CNN_Sotres_ResNet50(1, sequenceLength, 3, 3, 1);
% saveName = "Sotres2022_ResNet50-1D.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, false, true)

% Sotres ResNet50 without augmentation (original)
layers = CNN_Sotres_ResNet50(1, sequenceLength, 3, 3, 1);
saveName = "Sotres2022_ResNet50-1D_no_augmentation_100_epochs.mat";
general_CNN_evaluation_function_v1_100epochs(layers,preprocessedDataFile, saveName, false, false)
%%
% % Sotres ResNet50 with expanded features
% layers = CNN_Sotres_ResNet50(nFeatures, sequenceLength, 3, 3, 1);
% saveName = "Sotres2022_ResNet50-1D_expanded_features.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)
% 
% % Sotres ResNet50 with expanded features, no augmentation
% layers = CNN_Sotres_ResNet50(nFeatures, sequenceLength, 3, 3, 1);
% saveName = "Sotres2022_ResNet50-1D_expanded_features_no_augmentation.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, false)

% % CNN in sequence mode, 1 Conv block, LSTM sequence mode, with ReLu before
% layers = CNN_custom_pooling_after_lstm_relu(nFeatures, sequenceLength, 7);
% saveName = "one_conv_LSTM_sequence_relu.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)
% 
% % CNN in sequence mode, 1 Conv block, LSTM sequence mode, with ReLu before
% layers = CNN_custom_pooling_after_lstm_relu(nFeatures, sequenceLength, 7);
% saveName = "one_conv_LSTM_sequence_relu_no_augmentation.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, false)
% 
% % CNN in sequence mode, 2 Conv block, LSTM sequence mode
% layers = CNN_custom_pooling_after_lstm_2conv(nFeatures, sequenceLength, 7);
% saveName = "two_conv_LSTM_sequence_pooling.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)
%%
% % CNN 2 Conv block, LSTM sequence mode, relu
% layers = CNN_custom_pooling_after_lstm_2conv_relu(nFeatures, sequenceLength, 7);
% saveName = "two_conv_LSTM_sequence_pooling_relu.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)

% CNN 2 Conv block, LSTM sequence mode, relu, no augmentation
layers = CNN_custom_pooling_after_lstm_2conv_relu(nFeatures, sequenceLength, 7);
saveName = "two_conv_LSTM_sequence_pooling_relu_no_augmentation_100epochs.mat";
general_CNN_evaluation_function_v1_100epochs(layers,preprocessedDataFile, saveName, true, false)

% % CNN with 2 stacked bilstms and ReLu before final FC layer
% layers = CNN_stacked_bilstms_relu(nFeatures, sequenceLength, 7);
% saveName = "stacked_biLSTM_ReLu.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, true)
% 
% % CNN with 2 stacked bilstms and ReLu before final FC layer
% layers = CNN_stacked_bilstms_relu(nFeatures, sequenceLength, 7);
% saveName = "stacked_biLSTM_ReLu_no_augmentation.mat";
% general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, true, false)

% CNN 2 Conv block, biLSTM sequence mode, relu, no augmentation
layers = CNN_custom_pooling_after_bilstm_2conv_relu(1, sequenceLength, 7);
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_100epoch_onefeature.mat";
general_CNN_evaluation_function_v1_100epochs(layers,preprocessedDataFile, saveName, false, false)

%% Now for trimmed stuff

%% General parameters
sequenceLength = 2000;
nFeatures = 6;
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmNoTrimming.mat"; 
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo15nm.mat";
%preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_podocytes.mat";
% Path to helper functions
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder)

layers = CNN_custom_pooling_after_bilstm_2conv_relu(6, sequenceLength, 7);
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo15nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 20 nm
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo20nm.mat";
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo20nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 25 nm
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo25nm.mat";
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo25nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 30 nm
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo30nm.mat";
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo30nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 35 nm
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo35nm.mat";
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo35nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 40 nm
preprocessedDataFileTrimmed = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_EverythingOver40nmTrimmedTo40nm.mat";
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmTrimmedTo40nm.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile,preprocessedDataFileTrimmed, saveName, true, false)

% 40 nm
saveName = "two_conv_biLSTM_sequence_pooling_relu_no_augmentation_EverythingOver40nmNoTrimming.mat";
general_CNN_evaluation_function_v1_trimDeflTo20nm(layers,preprocessedDataFile ,preprocessedDataFile, saveName, true, false)