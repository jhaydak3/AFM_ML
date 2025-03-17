%% Script to train CNN model 10 times and record training durations

clc;
clear;
close all;

%% Parameters
sequenceLength = 2000;
nFeatures = 6;
preprocessedDataFile = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\training\\regression_processed_files\\processed_features_for_regression_All.mat";
helperFunctionsFolder = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\helperFunctions";
addpath(helperFunctionsFolder);

useAllFeatures = true;
useAugmentation = false;
repetitions = 10;
saveFileName = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\regressionTrainingTimes.mat";

%% Define CNN architecture
layers = CNN_custom_pooling_after_bilstm_2conv_relu(nFeatures, sequenceLength, 7);

%% Run training repeatedly and record times
trainingTimes = zeros(repetitions, 1);

for i = 1:repetitions
    fprintf('\n--- Starting training iteration %d/%d ---\n', i, repetitions);

    tic;
    [finalMSE, finalMAE, trainedNet] = train_CNN_model_regression_function(layers, preprocessedDataFile, "tempModel.mat", useAllFeatures, useAugmentation);
    elapsedTime = toc;

    trainingTimes(i) = elapsedTime;

    fprintf('Iteration %d completed in %.2f seconds.\n', i, elapsedTime);
end

%% Save training times
save(saveFileName, 'trainingTimes');
fprintf('All training times saved to "%s".\n', saveFileName);
