sequenceLength = 2000;
nFeatures = 6;
preprocessedDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_spherical_tissue.mat"; 
oldModel = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedRegressionModels\pooling_after_bilstm_2conv_relu.mat";
% Path to helper functions
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions";
addpath(helperFunctionsFolder)

layers = CNN_custom_pooling_after_bilstm_2conv_relu(1, sequenceLength, 7);
saveName = "spherical_test.mat";
[meanMSE, meanMAE, meanTrainMSE, meanTrainMAE] = fineTuneEvaluation(layers, preprocessedDataFile, oldModel, saveName, 0);