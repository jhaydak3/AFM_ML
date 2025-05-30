%% Script to calculate prediction times for CNN classification model

clc;
clear;
close all;

%% Parameters
numCurvesToPredict = 1000;
N = 1000;
sequenceLength = 2000; % Sequence length expected by the model

preprocessedDataFile = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\training\\classification_processed_files\\processed_features_for_classification_All.mat";
trainedModelFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedClassificationModels\pooling_after_bilstm_2conv_relu_classification.mat";
saveFileName = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\classificationPredictionTimes2.mat";

%% Load data
fprintf('Loading data from "%s"...\n', preprocessedDataFile);
data = load(preprocessedDataFile);
X = data.X;
numSamples = size(X, 3);

%% Load trained model
fprintf('Loading trained model from "%s"...\n', trainedModelFile);
loadedModel = load(trainedModelFile);
trainedNet = loadedModel.trainedNet;

%% Perform prediction repeatedly and record times
predictionTimes = zeros(N, 1);
% First iteration takes longer for some reason
for i = 1:N+1
    fprintf('Prediction iteration %d/%d...\n', i, N);

    % Select random curves
    randomIndices = randperm(numSamples, numCurvesToPredict);
    XEval = X(:, :, randomIndices);

    % Prepare data for prediction
    XEvalPerm = permute(XEval, [1, 3, 2]);
    dlXEval = dlarray(XEvalPerm, 'CBT');

    % Time prediction

    tic;
    predict(trainedNet, dlXEval);
    elapsedTime = toc;
    if i > 1
        predictionTimes(i-1) = elapsedTime;
    end

    fprintf('Iteration %d prediction completed in %.4f seconds.\n', i, elapsedTime);
end

%% Save prediction times
save(saveFileName, 'predictionTimes');
fprintf('All prediction times saved to "%s".\n', saveFileName);