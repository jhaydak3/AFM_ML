function results = fineTuneEvaluationFunctionOfN(layers, preprocessedDataFile, oldModel, saveName, useAugmentation)
% fineTuneEvaluationFunctionOfN Evaluate fine-tuning performance as a function
% of the number of training curves.
%
%   results = fineTuneEvaluationFunctionOfN(layers, preprocessedDataFile, oldModel, saveName, useAugmentation)
%
%   This function fine-tunes a pretrained CNN regression model for various numbers of
%   training curves and evaluates its performance on a fixed hold-out dataset.
%   A predetermined number of samples (numberHoldOut) are reserved exclusively for
%   evaluation. For each value specified in curveTrainSizes, the function randomly
%   selects that number of curves from the remaining pool, fine tunes the model (if the
%   training size is greater than 0) using trainModelCoreFineTuning, and then evaluates
%   the fine-tuned model on the hold-out set to calculate the Mean Absolute Error (MAE).
%   The procedure is repeated nBootstraps times for each training size.
%
%   Inputs:
%       layers             - CNN layers specification.
%       preprocessedDataFile - Full path to the .mat file containing the preprocessed data.
%       oldModel           - Pretrained model (e.g., a dlnetwork) to be fine tuned.
%       saveName           - File name to save the evaluation results.
%       useAugmentation    - Boolean flag; if true, augment the training data using augmentData.
%
%   Output:
%       results            - Structure array containing evaluation results for each
%                            combination of training set size and bootstrap iteration.
%
%   Note: No plotting is done in this function. The goal is to create data suitable for
%         later plotting of MAE vs. the number of curves used in fine-tuning.
%
%   Example:
%       results = fineTuneEvaluationFunctionOfN(layers, 'preprocessedData.mat', ...
%                        'oldModel.mat', 'evaluationResults.mat', true);

    %% =============== Parameter Settings and Seed Initialization =============== %%
    % Evaluation parameters
    numberHoldOut   = 100;         % Number of curves reserved for hold-out evaluation.
    curveTrainSizes = 0:25:250;     % Vector specifying training sizes for fine tuning.
    nBootstraps     = 10;           % Number of bootstrap iterations per training size.
    pad             = 100;         % Padding parameter for data augmentation.
    seedNumber      = 1337;        % Seed for reproducibility.
    rng(seedNumber);

    %% ==================== Load the Pretrained Model if Needed ==================== %%
    if ischar(oldModel) || isstring(oldModel)
        fprintf('Loading pretrained model from file: %s\n', oldModel);
        tmp = load(oldModel);
        if isfield(tmp, 'net')
            oldModel = tmp.net;
        elseif isfield(tmp, 'trainedNet')
            oldModel = tmp.trainedNet;
        else
            error('The loaded file does not contain a recognized network variable (net or trainedNet).');
        end
    end

    %% ==================== Load the Preprocessed Data ==================== %%
    fprintf('Loading preprocessed data from "%s"...\n', preprocessedDataFile);
    data = load(preprocessedDataFile);
    requiredFieldsPD = {'X', 'Y', 'maxExtValues', 'minExtValues', 'rawExt', 'rawDefl', ...
                        'fileIndices', 'fileRow', 'fileCol', 'b','th','R','v','spring_constant','goodOrBad'};
    for i = 1:length(requiredFieldsPD)
        if ~isfield(data, requiredFieldsPD{i})
            error('Field "%s" is missing from "%s".', requiredFieldsPD{i}, preprocessedDataFile);
        end
    end

    X = data.X;           % [numFeatures x sequenceLength x numSamples]
    Y = data.Y;           % [numSamples x 1]
    % Extract conversion parameters to physical units (nm)
    maxExtValues = data.maxExtValues';
    minExtValues = data.minExtValues';
    
    numSamples = size(X, 3);
    fprintf('Loaded data with %d samples.\n', numSamples);

    if numSamples <= numberHoldOut
        error('Number of samples (%d) must be greater than numberHoldOut (%d).', numSamples, numberHoldOut);
    end

    %% ==================== Split Data into Hold-Out and Training Pool ==================== %%
    indices = randperm(numSamples);
    holdoutIndices = indices(1:numberHoldOut);
    trainingPoolIndices = indices(numberHoldOut+1:end);

    XHoldout = X(:, :, holdoutIndices);
    YHoldout = Y(holdoutIndices);

    fprintf('Hold-out set: %d samples. Training pool: %d samples.\n', numberHoldOut, numel(trainingPoolIndices));

    %% ==================== Fine-Tuning Across Different Training Sizes ==================== %%
    totalIterations = numel(curveTrainSizes) * nBootstraps;
    results = struct('curveTrainSize', [], 'bootstrapIndex', [], 'trainingIndices', [], ...
                     'timeToTrain', [], 'holdoutMAE', [], 'holdoutMAE_nm', [], ...
                     'YPredHoldout', [], 'YHoldout', []);
    resultCount = 1;

    for i = 1:length(curveTrainSizes)
        currentSize = curveTrainSizes(i);
        fprintf('Evaluating for curve training size = %d\n', currentSize);
        for j = 1:nBootstraps
            fprintf('  Bootstrap iteration %d of %d...\n', j, nBootstraps);
            if currentSize > 0
                if currentSize > numel(trainingPoolIndices)
                    error('Requested training size (%d) exceeds available training pool size (%d).', currentSize, numel(trainingPoolIndices));
                end
                % Randomly select training curves from the training pool
                trainingIndices = randsample(trainingPoolIndices, currentSize);
                XTrain = X(:, :, trainingIndices);
                YTrain = Y(trainingIndices);

                % Augment training data if enabled
                if useAugmentation
                    fprintf('    Augmenting training data...\n');
                    [XTrainAug, YTrainAug] = augmentData(XTrain, YTrain, pad);
                else
                    XTrainAug = XTrain;
                    YTrainAug = YTrain;
                end

                % Fine-tune the model starting from the pretrained model
                fprintf('    Fine tuning the network...\n');
                try
                    tic;
                    trainedNet = trainModelCoreFineTuning(oldModel, XTrainAug, YTrainAug);
                    timeElapsed = toc;
                    fprintf('    Fine tuning completed in %.2f seconds.\n', timeElapsed);
                catch ME
                    error('Error during fine tuning for training size %d, bootstrap %d: %s', currentSize, j, ME.message);
                end
            else
                % For training size 0, do not fine-tune; use the original model.
                fprintf('    Using original model without fine tuning.\n');
                trainedNet = oldModel;
                timeElapsed = 0;
                trainingIndices = [];
            end

            %% ==================== Evaluate on the Hold-Out Set ==================== %%
            % Prepare hold-out data for prediction: permute dimensions to [C x B x T]
            XHoldoutForPred = permute(XHoldout, [1, 3, 2]);
            XHoldoutForPred = dlarray(XHoldoutForPred, 'CBT');
            try
                YPredHoldout = extractdata(predict(trainedNet, XHoldoutForPred))';
            catch ME
                warning('Error during hold-out prediction for training size %d, bootstrap %d: %s', currentSize, j, ME.message);
                YPredHoldout = NaN(size(YHoldout));
            end

            % Calculate Mean Absolute Error (MAE) on the hold-out set (normalized units)
            holdoutMAE = mean(abs(YPredHoldout - YHoldout));
            
            % Convert hold-out predictions and ground-truth to physical units (nm)
            maxHoldOut = maxExtValues(holdoutIndices);
            minHoldOut = minExtValues(holdoutIndices);
            predHoldout_nm = YPredHoldout .* (maxHoldOut - minHoldOut) + minHoldOut;
            trueHoldout_nm = YHoldout .* (maxHoldOut - minHoldOut) + minHoldOut;
            holdoutMAE_nm = mean(abs(predHoldout_nm - trueHoldout_nm));

            %% ==================== Store Results ==================== %%
            results(resultCount).curveTrainSize = currentSize;
            results(resultCount).bootstrapIndex = j;
            results(resultCount).trainingIndices = trainingIndices;
            results(resultCount).timeToTrain = timeElapsed;
            results(resultCount).holdoutMAE = holdoutMAE;
            results(resultCount).holdoutMAE_nm = holdoutMAE_nm;
            results(resultCount).YPredHoldout = YPredHoldout;
            results(resultCount).YHoldout = YHoldout;
            resultCount = resultCount + 1;
        end
    end

    %% ==================== Save the Collected Results ==================== %%
    fprintf('Saving evaluation results to "%s"...\n', saveName);
    save(saveName, 'results', 'numberHoldOut', 'curveTrainSizes', 'nBootstraps', ...
         'holdoutIndices', 'trainingPoolIndices', 'seedNumber');
    fprintf('Results saved successfully.\n');
end
