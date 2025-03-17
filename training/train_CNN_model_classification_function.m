function train_CNN_model_classification_function(layers, preprocessedDataFile, saveName, useAllFeatures)
% train_CNN_model_classification_function trains a custom CNN model on the entire dataset,
% saves the trained network and training info, and saves a plot of the ROC curve with
% annotations of different threshold values.
%
%   train_CNN_model_classification_function(layers, preprocessedDataFile, saveName, useAllFeatures)
%
%   INPUTS:
%       layers              - Layer array for the custom CNN.
%       preprocessedDataFile- .mat file containing preprocessed data (must include fields "X" and "goodOrBad")
%       saveName            - .mat filename where the trained network and training info will be saved.
%       useAllFeatures      - If true, uses all features in X; otherwise, uses only the first feature.
%
%   OUTPUT:
%       None. The trained model (and training info) and a ROC plot are saved to disk.
%
%   Example:
%       layers = [ ... ]; % define your CNN layers
%       train_CNN_model_classification_function(layers, 'myPreprocessedData.mat', 'trainedModel.mat', true);

    clc;
    close all;
    fprintf('Starting training on all data using the custom CNN model...\n');

    %% Set random seed for reproducibility
    rng(1337, 'twister');

    %% (Optional) Add helper functions folder if needed
    helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v4_CNI_predict\helperFunctions";
    addpath(helperFunctionsFolder);

    %% Load Data
    fprintf('Loading preprocessed data from "%s"...\n', preprocessedDataFile);
    data = load(preprocessedDataFile);

    % Check that required fields exist
    requiredFields = {'X', 'goodOrBad'};
    for i = 1:numel(requiredFields)
        if ~isfield(data, requiredFields{i})
            error('Field "%s" missing in "%s".', requiredFields{i}, preprocessedDataFile);
        end
    end

    X = data.X;
    goodOrBad = data.goodOrBad;  % can be numeric or string-based

    numSamples = size(X, 3);
    fprintf('Loaded %d samples.\n', numSamples);

    %% Convert Labels to Categorical ('reject' and 'accept')
    if isnumeric(goodOrBad)
        labelsStr = strings(size(goodOrBad));
        labelsStr(goodOrBad == 0) = "reject";
        labelsStr(goodOrBad == 1) = "accept";
    else
        % Handle string or char arrays: convert '0' to "reject", etc.
        labelsStr = strings(size(goodOrBad));
        labelsStr(goodOrBad == '0') = "reject";
        labelsStr(goodOrBad == '1') = "accept";
    end
    labelsCat = categorical(labelsStr, {'reject', 'accept'});

    %% Select Features
    if useAllFeatures
        XData = X;
    else
        XData = X(1,:,:);
    end

    %% Prepare Data for Training
    % Permute XData so that the dimensions match what the network expects.
    % (Original data is assumed to be arranged as [features, time, samples].)
    XPerm = permute(XData, [1, 3, 2]);
    dlX = dlarray(XPerm, 'CBT');  % 'C': channel, 'B': batch, 'T': time (or other feature)

    % Shuffle the data
    rp = randperm(size(dlX, 2));
    dlX = dlX(:, rp, :);
    labelsCat = labelsCat(rp);

    %% Define the CNN and Training Options
    net = dlnetwork(layers);
    
    % Note: Adjust the trainingOptions as needed for your application.
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none', ...  % Change to 'training-progress' to see training progress
        'ValidationFrequency', 50, ...
        'InitialLearnRate', 1e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 1, ...
        'LearnRateDropPeriod', 1, ...
        'Metrics', ["precision", "accuracy", "auc", "fscore", "recall"], ...
        'ObjectiveMetricName', "auc", ...
        'OutputNetwork', 'last-iteration');

    %% Train the Network
    fprintf('Training on the full dataset...\n');
    try
        % Note: The helper function "trainnet" is assumed to be available.
        [trainedNet, info] = trainnet(dlX, labelsCat', net, "crossentropy", options);
        fprintf('Training completed successfully.\n');
    catch ME
        error('Error during training: %s', ME.message);
    end

    %% Save the Trained Model and Training Info
    fprintf('Saving trained model to "%s"...\n', saveName);
    save(saveName, 'trainedNet', 'info', 'layers', 'options');
    fprintf('Model saved successfully.\n');

    %% Compute Predictions on the Training Data for the ROC Curve
    % Here we assume that the network outputs a 2-by-N array, with row 1 corresponding
    % to the probability of the "reject" class.
    YPred = predict(trainedNet, dlX);
    YPred = extractdata(YPred);  % convert from dlarray to numeric array

    %% Compute the ROC Curve
    % The "perfcurve" function computes the false positive rate (fpr), true positive rate (tpr)
    % and threshold values. We use "reject" as the positive class.
    [fpr, tpr, thr, auc] = perfcurve(labelsCat, YPred(1,:)', 'reject');

    %% Plot the ROC Curve with Annotations for Thresholds
    figure;
    plot(fpr, tpr, 'b-', 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve (AUC = %.3f)', auc));
    grid on;
    hold on;
    
    % Annotate a subset of the points with threshold values.
    % (Annotating every point can clutter the figure.)
    numPoints = numel(thr);
    sampleCount = min(10, numPoints);  % annotate up to 10 points
    sampleIdx = round(linspace(1, numPoints, sampleCount));
    for i = 1:length(sampleIdx)
        idx = sampleIdx(i);
        txt = sprintf('%.2f', thr(idx));
        % Adjust the text position as needed using 'HorizontalAlignment' and 'VerticalAlignment'
        text(fpr(idx), tpr(idx), txt, 'FontSize', 8, 'HorizontalAlignment','right', 'VerticalAlignment','bottom');
    end

    % Identify and highlight the best threshold (minimizing distance to the point (0,1))
    dist2Corner = sqrt((fpr - 0).^2 + (tpr - 1).^2);
    [~, bestIdx] = min(dist2Corner);
    plot(fpr(bestIdx), tpr(bestIdx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(fpr(bestIdx), tpr(bestIdx), sprintf('Best: %.2f', thr(bestIdx)), 'FontSize', 10, 'Color', 'r', ...
         'HorizontalAlignment','left', 'VerticalAlignment','top');
    hold off;
    
    %% Save the ROC Plot to a PNG File
    % Create a file name based on the "saveName" (which is assumed to be a .mat file)
    [~, name, ~] = fileparts(saveName);
    rocPlotName = strcat(name, '_ROC.png');
    saveas(gcf, rocPlotName);
    fprintf('ROC plot saved as "%s".\n', rocPlotName);
end
