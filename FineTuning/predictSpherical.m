%% predictContactPoint.m
% Clear workspace and close figures
clear; close all; clc;

%% Specify file paths
modelFile = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedRegressionModels\pooling_after_bilstm_2conv_relu.mat';
dataFile  = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_spherical_tissue.mat';

%% Load the trained model
modelData = load(modelFile);
if isfield(modelData, 'net')
    net = modelData.net;
elseif isfield(modelData, 'trainedNet')
    net = modelData.trainedNet;
else
    error('Model file does not contain a recognized network variable.');
end

%% Load preprocessed data
data = load(dataFile);
requiredFields = {'X', 'Y', 'minExtValues', 'maxExtValues'};
for i = 1:length(requiredFields)
    if ~isfield(data, requiredFields{i})
        error('Field "%s" is missing from the data file.', requiredFields{i});
    end
end

X         = data.X;            % Expected dimensions: [sequenceLength x 6 x numSamples]
Y_norm    = data.Y;            % Normalized ground truth contact points, [numSamples x 1]
minExt    = data.minExtValues; % Either a scalar or a vector of length numSamples
maxExt    = data.maxExtValues; % Either a scalar or a vector of length numSamples

%% Adjust the shape of X for the network
% The network expects input with dimensions: [channels x sequenceLength x numSamples]
X_reshaped = permute(X, [2, 1, 3]);  % Now X_reshaped is [6 x sequenceLength x numSamples]

%% Make predictions using the trained model
YPred_norm = predict(net, X_reshaped);
YPred_norm = squeeze(YPred_norm);

% If the predictions have more than one value per sample, select the final prediction.
% (Modify this reduction if your desired prediction is computed differently.)
if size(YPred_norm, 1) > 1
    YPred_norm = YPred_norm(end, :);
end

%% Convert normalized predictions and ground truth to physical units (nm)
if isvector(minExt) && (length(minExt) == length(Y_norm))
    YPred_nm = YPred_norm .* (maxExt - minExt) + minExt;
    Y_true_nm = Y_norm    .* (maxExt - minExt) + minExt;
else
    % If minExt and maxExt are scalars
    YPred_nm = YPred_norm .* (maxExt - minExt) + minExt;
    Y_true_nm = Y_norm    .* (maxExt - minExt) + minExt;
end

%% Compute the absolute error (in nm)
absError_nm = abs(YPred_nm - Y_true_nm);

%% Calculate overall error statistics
meanError   = mean(absError_nm(:));
medianError = median(absError_nm(:));
stdError    = std(absError_nm(:));

%% Plot histogram of absolute error
figure;
histogram(absError_nm, 'BinWidth', 1);  % Adjust the bin width if needed
xlabel('Absolute Error (nm)');
ylabel('Frequency');
title('Histogram of Contact Point Absolute Error (nm)');

%% Print error statistics (only once)
fprintf('Mean Absolute Error: %.2f nm\n', meanError);
fprintf('Median Absolute Error: %.2f nm\n', medianError);
fprintf('Standard Deviation: %.2f nm\n', stdError);
