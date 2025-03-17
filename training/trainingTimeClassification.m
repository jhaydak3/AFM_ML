%% Script to repeatedly train CNN classification model and record training durations

clc;
clear;
close all;

%% Parameters
sequenceLength = 2000;
nFeatures = 6;
repetitions = 10;
preprocessedDataFile = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\training\\classification_processed_files\\processed_features_for_classification_All.mat";
saveFileName = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\classificationTrainingTimes.mat";
helperFunctionsFolder = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\helperFunctions";

addpath(helperFunctionsFolder);

%% Define CNN architecture
layers = CNN_custom_pooling_after_lstm_2conv_relu_classification(nFeatures, sequenceLength, 7);

%% Perform training repeatedly and record times
trainingTimes = zeros(repetitions, 1);

for i = 1:repetitions
    fprintf('\n--- Starting training iteration %d/%d ---\n', i, repetitions);

    tic;
    train_CNN_model_classification_function(layers, preprocessedDataFile, "tempClassificationModel.mat", true);
    elapsedTime = toc;

    trainingTimes(i) = elapsedTime;

    fprintf('Iteration %d completed in %.2f seconds.\n', i, elapsedTime);
end

%% Save training times
save(saveFileName, 'trainingTimes');
fprintf('All training times saved to "%s".\n', saveFileName);
