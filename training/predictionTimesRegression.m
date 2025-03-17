%% Script to evaluate CNN regression model on random curves repeatedly

clc;
clear;
close all;

%% Parameters
numCurvesToPredict = 1000;
N = 1000;
preprocessedDataFile = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\training\\regression_processed_files\\processed_features_for_regression_All.mat";
trainedModelFile ="C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedClassificationModels\two_conv_LSTM_sequence_pooling_relu_classification.mat";
saveFileName = "C:\\Users\\MrBes\\Documents\\MATLAB\\AFM_ML\\AFM_ML_v6_sandbox\\evaluationTimeRegression.mat";

%% Load data
fprintf('Loading data from "%s"...\n', preprocessedDataFile);
data = load(preprocessedDataFile);
X = data.X;
numSamples = size(X, 3);

%% Load trained model
fprintf('Loading trained model from "%s"...\n', trainedModelFile);
modelData = load(trainedModelFile);
trainedNet = modelData.trainedNet;

%% Evaluate repeatedly and record prediction times
predictionTimes = zeros(N, 1);

for i = 1:N
    fprintf('Evaluation iteration %d/%d...\n', i, N);

    % Select random curves
    randomIndices = randperm(numSamples, numCurvesToPredict);
    XEval = X(:, :, randomIndices);

    % Prepare data for prediction
    XEvalForPred = permute(XEval, [1, 3, 2]);
    XEvalForPred = dlarray(XEvalForPred, 'CBT');

    % Measure prediction time
    tic;
    YPred = predict(trainedNet, XEvalForPred);
    elapsedTime = toc;

    predictionTimes(i) = elapsedTime;

    fprintf('Iteration %d prediction completed in %.4f seconds.\n', i, elapsedTime);
end

%% Save prediction times
save(saveFileName, 'predictionTimes');
fprintf('All prediction times saved to "%s".\n', saveFileName);
