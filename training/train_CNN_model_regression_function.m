function [finalMSE, finalMAE, trainedNet] = train_CNN_model_regression_function(layers, preprocessedDataFile, saveModelName, useAllFeatures, useAugmentation)
% train_CNN_model
% Trains a CNN model on all available data and saves the trained model.
%
% Inputs:
%   layers              - Defined CNN architecture
%   preprocessedDataFile - Path to preprocessed .mat file containing features & labels
%   saveModelName       - Filename to save trained model
%   useAllFeatures      - Boolean flag: true to use all features, false to use only raw deflection
%   useAugmentation     - Boolean flag: true to augment data, false to train on raw data only
%
% Outputs:
%   finalMSE            - Mean Squared Error on training data
%   finalMAE            - Mean Absolute Error on training data
%   trainedNet          - Trained CNN model

clc;
close all;
fprintf('Starting CNN training on all available data...\n');

%% ==================== Load the Preprocessed Data ==================== %%
fprintf('Loading preprocessed data from "%s"...\n', preprocessedDataFile);
data = load(preprocessedDataFile);

% Validate loaded data
requiredFields = {'X', 'Y', 'maxExtValues', 'minExtValues', 'rawExt', 'rawDefl', ...
                  'b', 'th', 'R', 'v', 'spring_constant', 'goodOrBad'};
for i = 1:length(requiredFields)
    if ~isfield(data, requiredFields{i})
        error('Field "%s" is missing from "%s".', requiredFields{i}, preprocessedDataFile);
    end
end

X = data.X; % [numFeatures x sequenceLength x numSamples]
Y = data.Y; % [numSamples x 1]
maxExtValues = data.maxExtValues';
minExtValues = data.minExtValues';

numSamples = size(X, 3);
fprintf('Loaded data with %d samples.\n', numSamples);

%% ==================== Prepare Training Data ==================== %%
if useAllFeatures
    XTrain = X;  % Use all features
else
    XTrain = X(1, :, :);  % Use only raw deflection curve
end
YTrain = Y;

fprintf('Using %d training samples.\n', numSamples);

%% ==================== Data Augmentation (if enabled) ==================== %%
if useAugmentation
    fprintf('Augmenting training data...\n');
    pad = 100; % Padding for augmentation
    [XTrain, YTrain] = augmentData(XTrain, YTrain, pad);
end

%% ==================== Define the CNN Model ==================== %%
fprintf('Defining CNN architecture...\n');
net = dlnetwork(layers);

%% ==================== Train the Model ==================== %%
fprintf('Starting CNN training...\n');
try
    tic;
    trainedNet = trainModelCore(layers, XTrain, YTrain);
    trainingTime = toc;
    fprintf('Training completed successfully in %.2f seconds.\n', trainingTime);
catch ME
    disp(getReport(ME, 'extended'));
    error('Error during training: %s', ME.message);
end

%% ==================== Evaluate Model Performance ==================== %%
fprintf('Evaluating trained model performance...\n');

% Prepare input data for prediction
XTrainForPred = permute(XTrain, [1, 3, 2]); % Reshape [C x B x T]
XTrainForPred = dlarray(XTrainForPred, 'CBT');

% Make predictions
YPredTrain = predict(trainedNet, XTrainForPred);
YPredTrain = extractdata(YPredTrain)';

% Compute errors
trainErrors = YPredTrain - YTrain;
finalMSE = mean(trainErrors.^2);
finalMAE = mean(abs(trainErrors));

% Convert back to nm
predTrainNm = YPredTrain .* (maxExtValues - minExtValues) + minExtValues;
trueTrainNm = YTrain .* (maxExtValues - minExtValues) + minExtValues;
trainErrorsNm = predTrainNm - trueTrainNm;

finalMSE_nm = mean(trainErrorsNm.^2);
finalMAE_nm = mean(abs(trainErrorsNm));

%% ==================== Save Trained Model ==================== %%
fprintf('Saving trained model to "%s"...\n', saveModelName);
save(saveModelName, 'trainedNet', 'finalMSE', 'finalMAE', 'finalMSE_nm', 'finalMAE_nm');
fprintf('Model saved successfully.\n');

%% ==================== Print Summary ==================== %%
fprintf('\n=== Training Complete ===\n');
fprintf('Mean Squared Error (MSE): %.6f\n', finalMSE);
fprintf('Mean Absolute Error (MAE): %.6f\n', finalMAE);
fprintf('Mean Squared Error (MSE, nm): %.6f nm^2\n', finalMSE_nm);
fprintf('Mean Absolute Error (MAE, nm): %.6f nm\n', finalMAE_nm);

end
