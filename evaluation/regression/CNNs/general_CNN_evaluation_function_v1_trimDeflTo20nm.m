function [meanMSE, meanMAE, meanTrainMSE, meanTrainMAE] = general_CNN_evaluation_function_v1_trimDeflTo20nm(layers, preprocessedDataFile, preprocessedDataFileTrimmed, saveName, useAllFeatures, useAugmentation)
%EVALUATE_MODEL_REGRESSION_CNN_CUSTOM_REGRESSION
% Performs k-fold cross-validation for regression using a custom CNN model.
% but with TRIMMED data in the testing portion (limited deflection curves at ~20 nm).
%
% Calculates regression metrics for Contact Points (CP) and defers Modulus 
% (Hertzian and 500 nm) calculations until after the k-fold loop.
% Logs bad predictions exceeding a specified error threshold.
%
% Outputs:
%   meanMSE       - Mean Squared Error over k folds (Test Data, contact point in normalized scale)
%   meanMAE       - Mean Absolute Error over k folds (Test Data, contact point in normalized scale)
%   meanTrainMSE  - Mean Squared Error over k folds (Training Data, contact point in normalized scale)
%   meanTrainMAE  - Mean Absolute Error over k folds (Training Data, contact point in normalized scale)
%
% Saves:
%   'crossoverResults_regression_CNN_custom.mat' containing predictions and indices
%   Scatter plots in 'Evaluation_Figures' directory
%   'bad_predictions.txt' logging predictions exceeding the error threshold
%
% Important difference from the original version:
%   The TRAINING data is loaded from preprocessedDataFile (full depth),
%   while the TEST data is loaded from preprocessedDataFileTrimmed (20 nm).
%
% If useAllFeatures is true, uses expanded. If false, uses only raw deflection curve.

clc;
close all;
fprintf('Starting k-fold cross-validation using Custom CNN Regression Model with Trimmed Test Data (20nm)...\\n');

%% ==================== Configuration Parameters ==================== %%
k = 5; 
pad = 100; 
filter1 = 7; 
filter2 = 7; 
errorThreshold_kPa = 7.5;
seedNumber = 1337;
rng(seedNumber);
indentationDepth_nm = 500; % nm
testOnlyOnGood = false;

%% ==================== Load Original and Trimmed Data ==================== %%
fprintf('Loading original preprocessed data from \"%s\"...\\n', preprocessedDataFile);
data = load(preprocessedDataFile);

fprintf('Loading trimmed preprocessed data from \"%s\"...\\n', preprocessedDataFileTrimmed);
dataTrimmed = load(preprocessedDataFileTrimmed);

% Validate loaded data (assuming same fields for both)
requiredFieldsPD = {'X', 'Y', 'maxExtValues', 'minExtValues', 'rawExt', 'rawDefl', ...
                    'fileIndices', 'fileRow', 'fileCol', ...
                    'b','th','R','v','spring_constant','goodOrBad'};
for i = 1:length(requiredFieldsPD)
    if ~isfield(data, requiredFieldsPD{i}) || ~isfield(dataTrimmed, requiredFieldsPD{i})
        error('Field \"%s\" is missing from one of the preprocessed files.', requiredFieldsPD{i});
    end
end



%% Original (Train) Data Extraction
X = data.X; % [numFeatures x sequenceLength x numSamples]
Y = data.Y; % [numSamples x 1]
maxExtValues = data.maxExtValues';
minExtValues = data.minExtValues';
goodOrBad = data.goodOrBad;
numSamples = size(X, 3);

fprintf('Loaded original data with %d samples.\\n', numSamples);

%% Trimmed (Test) Data Extraction
X_trimmed = dataTrimmed.X;
Y_trimmed = dataTrimmed.Y;
maxExtValues_trimmed = dataTrimmed.maxExtValues';
minExtValues_trimmed = dataTrimmed.minExtValues';


%% ==================== Initialize Performance Metrics ==================== %%
mseScores = zeros(k, 1);
maeScores = zeros(k, 1);
trainMseScores = zeros(k, 1);
trainMaeScores = zeros(k, 1);

mseScoresNm = zeros(k, 1);
maeScoresNm = zeros(k, 1);
trainMseScoresNm = zeros(k, 1);
trainMaeScoresNm = zeros(k, 1);

% Initialize cell arrays to store predictions and actual values
YTrain_all = cell(k, 1);
YTest_all = cell(k, 1);
YPredTrain_all = cell(k, 1);
YPredTest_all = cell(k, 1);

% Initialize structure array to store results
results = struct('fold', {}, ...
                 'trainIndices', {}, 'testIndices', {}, ...
                 'YTrain', {}, 'YPredTrain', {}, ...
                 'YTest', {}, 'YPredTest', {}, ...
                 'YTrainNm', {}, 'YPredTrainNm', {}, ...
                 'YTestNm', {}, 'YPredTestNm', {}, ...
                 'HertzianModulusActual_train', {}, ...
                 'HertzianModulusPredicted_train', {}, ...
                 'Modulus500nmActual_train', {}, ...
                 'Modulus500nmPredicted_train', {}, ...
                 'HertzianModulusActual_test', {}, ...
                 'HertzianModulusPredicted_test', {}, ...
                 'Modulus500nmActual_test', {}, ...
                 'Modulus500nmPredicted_test', {}, ...
                 'badHertzCountPred', {}, 'badHertzCountActual', {}, ...
                 'bad500CountPred', {}, 'bad500CountActual', {} , ...
                 'timeToTrain', {});

%% ==================== Define k-Fold Cross-Validation Indices ==================== %%
fprintf('Setting up %d-fold cross-validation...\\n', k);

indices = zeros(numSamples,1);
goodCurveIndices = goodOrBad == 1;
numGoodCurves = sum(goodCurveIndices);
indicesGood = crossvalind('Kfold',numGoodCurves,k);
indices(goodOrBad == 1) = indicesGood;

badCurveIndices = goodOrBad == 0;
numBadCurves = sum(badCurveIndices);
indicesBad = crossvalind('Kfold',numBadCurves,k);
indices(goodOrBad == 0) = indicesBad;

%% ==================== Perform k-Fold Cross-Validation ==================== %%
fprintf('Starting k-fold cross-validation with %d folds...\\n', k);

for fold = 1:k
    fprintf('Processing Fold %d of %d...\\n', fold, k);

    % Split data into training and testing based on current fold
    testIdx = (indices == fold);
    trainIdx = ~testIdx;

    % TRAINING from the original data
    if useAllFeatures
        XTrain = X(:, :, trainIdx);
    else
        XTrain = X(1, :, trainIdx);
    end
    YTrain = Y(trainIdx);

    % TEST from the trimmed data
    if useAllFeatures
        XTest = X_trimmed(:, :, testIdx);
    else
        XTest = X_trimmed(1, :, testIdx);
    end
    YTest = Y_trimmed(testIdx);

    fprintf('Fold %d: Training samples = %d | Testing samples (trimmed) = %d\n', fold, sum(trainIdx), sum(testIdx));

    %% ==================== Augment the Training Data ==================== %%
    if useAugmentation == true
        fprintf('Fold %d: Augmenting training data...\n', fold);
        [XTrainAug, YTrainAug] = augmentData(XTrain, YTrain, pad);
    else
        XTrainAug = XTrain;
        YTrainAug = YTrain;
    end

    %% ==================== Define the Regression Neural Network ==================== %%
    fprintf('Fold %d: Defining the CNN architecture...\\n', fold);
    % 'layers' is passed in; we assume it is a valid layerGraph or layer array
    % with filter1 and filter2 as needed. 
    % For example usage: layers = CNN_custom(size(XTrainAug,1), size(XTrainAug,2), filter1, filter2);

    net = dlnetwork(layers);

    %% ==================== Train the Network ==================== %%
    fprintf('Starting training of the neural network for fold %d...\n', fold);
    predictSecondWay = false;
    try
        tic
        % Train the network (user-provided helper)
        trainedNet = trainModelCore(layers, XTrainAug, YTrainAug);
        fprintf('Training completed successfully for fold %d.\\n', fold);
        timeElapsed = toc;
    catch ME
        disp(getReport(ME, 'extended')); 
        warning('Error during training for fold %d: %s. \n Trying with trainModelCore2.\n', fold, ME.message);
        try
            tic
            trainedNet = trainModelCore2(layers, XTrainAug, YTrainAug);
            timeElapsed = toc;
            predictSecondWay = true;
        catch ME2
            disp(getReport(ME2, 'extended'));
            error('Error during training for fold %d: %s.\n', fold, ME2.message);
        end
    end

    %% ==================== Predictions on Training & Testing Data ==================== %%
    if ~predictSecondWay
        % Training
        XTrainAugForPred = permute(XTrainAug, [1,3,2]); % [C x B x T]
        XTrainAugForPred = dlarray(XTrainAugForPred, 'CBT');
        fprintf('Fold %d: Making predictions on training data...\n', fold);
        YPredTrainAug = predict(trainedNet, XTrainAugForPred);  
        YPredTrainAug = extractdata(YPredTrainAug)';
        YPredTrain = YPredTrainAug;  % We aren't separating out the augmented portion here

        % Testing
        fprintf('Fold %d: Making predictions on testing data (trimmed)...\n', fold);
        XTestForPred = permute(XTest, [1,3,2]); 
        XTestForPred = dlarray(XTestForPred, 'CBT');

        try
            YPredTest = predict(trainedNet, XTestForPred);
            YPredTest = extractdata(YPredTest)';
        catch ME
            warning('Fold %d: Error during prediction on test data: %s. Skipping this fold.', fold, ME.message);
            mseScores(fold) = NaN;
            maeScores(fold) = NaN;
            mseScoresNm(fold) = NaN;
            maeScoresNm(fold) = NaN;
            continue;
        end
    else
        % If we used trainModelCore2
        numFeaturesTrain  = size(XTrainAug,1);
        sequenceLenTrain  = size(XTrainAug,2);
        batchSizeTrain    = size(XTrainAug,3);
        XTrainAugForPred = permute(XTrainAug, [2 4 1 3]);
        XTrainAugForPred = reshape(XTrainAugForPred, [sequenceLenTrain, 1, numFeaturesTrain, batchSizeTrain]);
        YPredTrain = predict(trainedNet,XTrainAugForPred);

        numFeaturesTest  = size(XTest,1);  
        sequenceLenTest  = size(XTest,2);
        batchSizeTest    = size(XTest,3);
        XTestForPred = permute(XTest, [2 4 1 3]);
        XTestForPred = reshape(XTestForPred, [sequenceLenTest, 1, numFeaturesTest, batchSizeTest]);
        YPredTest = predict(trainedNet, XTestForPred);
    end

    %% ==================== Compute CP Metrics (Normalized) ==================== %%
    trainErrorsNorm = YPredTrain - YTrain;
    trainMseScores(fold) = mean(trainErrorsNorm.^2);
    trainMaeScores(fold) = mean(abs(trainErrorsNorm));

    testErrorsNorm = YPredTest - YTest;
    mseScores(fold) = mean(testErrorsNorm.^2);
    maeScores(fold) = mean(abs(testErrorsNorm));

    %% ==================== Compute CP Metrics (Physical Units, nm) ==================== %%
    predTrainNm = YPredTrain .* (maxExtValues(trainIdx) - minExtValues(trainIdx)) + minExtValues(trainIdx);
    trueTrainNm = YTrain .* (maxExtValues(trainIdx) - minExtValues(trainIdx)) + minExtValues(trainIdx);
    trainErrorsNm = predTrainNm - trueTrainNm;
    trainMseScoresNm(fold) = mean(trainErrorsNm.^2);
    trainMaeScoresNm(fold) = mean(abs(trainErrorsNm));

    % ***Use TRIMMED values for the TEST portion***
    predTestNm = YPredTest .* (maxExtValues_trimmed(testIdx) - minExtValues_trimmed(testIdx)) + minExtValues_trimmed(testIdx);
    trueTestNm = YTest .* (maxExtValues_trimmed(testIdx) - minExtValues_trimmed(testIdx)) + minExtValues_trimmed(testIdx);
    testErrorsNm = predTestNm - trueTestNm;
    mseScoresNm(fold) = mean(testErrorsNm.^2);
    maeScoresNm(fold) = mean(abs(testErrorsNm));

    %% ==================== Store Predictions in Results Struct ==================== %%
    YTrain_all{fold} = YTrain;
    YTest_all{fold}  = YTest;
    YPredTrain_all{fold} = YPredTrain;
    YPredTest_all{fold}  = YPredTest;

    results(fold).fold = fold;
    results(fold).trainIndices = find(trainIdx);
    results(fold).testIndices  = find(testIdx);

    results(fold).YTrain   = YTrain;
    results(fold).YPredTrain = YPredTrain;
    results(fold).YTest    = YTest;
    results(fold).YPredTest= YPredTest;

    results(fold).YTrainNm     = trueTrainNm;
    results(fold).YPredTrainNm = predTrainNm;
    results(fold).YTestNm      = trueTestNm;
    results(fold).YPredTestNm  = predTestNm;

    results(fold).timeToTrain  = timeElapsed;

    fprintf('Fold %d: Completed.\\n', fold);
end

%% ==================== Calculate Moduli & Log Bad Predictions AFTER k-Fold ==================== %%
badPredictionsFile = 'bad_predictions.txt';
fileID = fopen(badPredictionsFile, 'w');  % Overwrite existing
if fileID == -1
    error('Cannot open file %s for writing.', badPredictionsFile);
else
    fprintf(fileID, 'Curves with Absolute Prediction Error > %.2f kPa\\n', errorThreshold_kPa);
    fprintf(fileID, 'Fold\\tCurveIdx\\tHertzianError(kPa)\\t500nmError(kPa)\\tFile\\tRow\\tCol\\n');
end

fprintf('Calculating modulus metrics and logging bad predictions...\\n');

for fold = 1:k
    fprintf('Fold %d: Calculating modulus metrics...\\n', fold);

    % Retrieve predictions and indices
    YTrain = results(fold).YTrain;
    YPredTrain = results(fold).YPredTrain;
    YTest = results(fold).YTest;
    YPredTest = results(fold).YPredTest;
    trainIdx = results(fold).trainIndices;
    testIdx  = results(fold).testIndices;

    if testOnlyOnGood
        % Evaluate only on good curves
        qualityTrain = goodOrBad(trainIdx);
        trainMask = logical(qualityTrain);
        trainIdxGood = trainIdx(trainMask);
        YTrainEval = YTrain(trainMask);
        YPredTrainEval = YPredTrain(trainMask);

        qualityTest = goodOrBad(testIdx);
        testMask = logical(qualityTest);
        testIdxGood = testIdx(testMask);
        YTestEval = YTest(testMask);
        YPredTestEval = YPredTest(testMask);

        % Calculate Moduli for Training (original data)
        [HertzianModulusActualTrain, HertzianModulusPredictedTrain, ...
            Modulus500nmActualTrain, Modulus500nmPredictedTrain] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTrainEval, YPredTrainEval, ...
            trainIdxGood, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);

        % Calculate Moduli for Testing (trimmed data)
        [HertzianModulusActualTest, HertzianModulusPredictedTest, ...
            Modulus500nmActualTest, Modulus500nmPredictedTest] = calculateModuli( ...
            dataTrimmed.rawExt, dataTrimmed.rawDefl, YTestEval, YPredTestEval, ...
            testIdxGood, minExtValues_trimmed, maxExtValues_trimmed, ...
            dataTrimmed.b, dataTrimmed.th, dataTrimmed.R, dataTrimmed.v, dataTrimmed.spring_constant, ...
            indentationDepth_nm);
    else
        % Calculate Moduli for Training (original data)
        [HertzianModulusActualTrain, HertzianModulusPredictedTrain, ...
            Modulus500nmActualTrain, Modulus500nmPredictedTrain] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTrain, YPredTrain, ...
            trainIdx, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);

        % Calculate Moduli for Testing (trimmed data)
        [HertzianModulusActualTest, HertzianModulusPredictedTest, ...
            Modulus500nmActualTest, Modulus500nmPredictedTest] = calculateModuli( ...
            dataTrimmed.rawExt, dataTrimmed.rawDefl, YTest, YPredTest, ...
            testIdx, minExtValues_trimmed, maxExtValues_trimmed, ...
            dataTrimmed.b, dataTrimmed.th, dataTrimmed.R, dataTrimmed.v, dataTrimmed.spring_constant, ...
            indentationDepth_nm);
    end

    badHertzCountActual = sum(isnan(HertzianModulusActualTest));
    bad500CountActual   = sum(isnan(Modulus500nmActualTest));
    badHertzCountPred   = sum(isnan(HertzianModulusPredictedTest));
    bad500CountPred     = sum(isnan(Modulus500nmPredictedTest));

    results(fold).HertzianModulusActual_train     = HertzianModulusActualTrain;
    results(fold).HertzianModulusPredicted_train  = HertzianModulusPredictedTrain;
    results(fold).Modulus500nmActual_train        = Modulus500nmActualTrain;
    results(fold).Modulus500nmPredicted_train     = Modulus500nmPredictedTrain;
    results(fold).HertzianModulusActual_test      = HertzianModulusActualTest;
    results(fold).HertzianModulusPredicted_test   = HertzianModulusPredictedTest;
    results(fold).Modulus500nmActual_test         = Modulus500nmActualTest;
    results(fold).Modulus500nmPredicted_test      = Modulus500nmPredictedTest;

    results(fold).badHertzCountActual            = badHertzCountActual;
    results(fold).badHertzCountPred              = badHertzCountPred;
    results(fold).bad500CountActual              = bad500CountActual;
    results(fold).bad500CountPred                = bad500CountPred;

    % Log bad predictions
    absErrorHertz = abs(HertzianModulusPredictedTest - HertzianModulusActualTest);
    absError500nm = abs(Modulus500nmPredictedTest - Modulus500nmActualTest);
    badIndicesHertz = find(absErrorHertz > errorThreshold_kPa);
    badIndices500nm = find(absError500nm > errorThreshold_kPa);
    badIndices = unique([badIndicesHertz; badIndices500nm]);

    for j = 1:length(badIndices)
        curveIdx = results(fold).testIndices(badIndices(j));
        fileName = data.fileIndices{curveIdx};
        row = data.fileRow(curveIdx);
        col = data.fileCol(curveIdx);

        fprintf(fileID, '%d\\t%d\\t%.3f\\t%.3f\\t%s\\t%d\\t%d\\n', ...
            fold, curveIdx, ...
            absErrorHertz(badIndices(j)), ...
            absError500nm(badIndices(j)), ...
            fileName, row, col);
    end
end
fclose(fileID);
fprintf('Bad predictions saved to \"%s\".\\n', badPredictionsFile);

%% ==================== Calculate Mean Performance Metrics Across All Folds ==================== %%
fprintf('Calculating mean performance metrics across all folds...\\n');
validFolds = ~isnan(mseScores) & ~isnan(maeScores) & ~isnan(mseScoresNm) & ~isnan(maeScoresNm);

meanMSE = mean(mseScores(validFolds));
meanMAE = mean(maeScores(validFolds));
meanTrainMSE = mean(trainMseScores(validFolds));
meanTrainMAE = mean(trainMaeScores(validFolds));

meanMSENm = mean(mseScoresNm(validFolds));
meanMAENm = mean(maeScoresNm(validFolds));
meanTrainMSENm = mean(trainMseScoresNm(validFolds));
meanTrainMAENm = mean(trainMaeScoresNm(validFolds));

fprintf('\\n=== Overall Performance Metrics Across All Folds ===\\n');
% Contact Points (Normalized)
fprintf('\\n--- Contact Points (Normalized) ---\\n');
fprintf('Mean Squared Error (MSE): %.6f\\n', meanMSE);
fprintf('Mean Absolute Error (MAE): %.6f\\n', meanMAE);
fprintf('Train Data - Mean Squared Error (MSE): %.6f\\n', meanTrainMSE);
fprintf('Train Data - Mean Absolute Error (MAE): %.6f\\n', meanTrainMAE);

% Contact Points (nm)
fprintf('\\n--- Contact Points (nm) ---\\n');
fprintf('Mean Squared Error (MSE): %.6f nm^2\\n', meanMSENm);
fprintf('Mean Absolute Error (MAE): %.6f nm\\n', meanMAENm);
fprintf('Train Data - Mean Squared Error (MSE): %.6f nm^2\\n', meanTrainMSENm);
fprintf('Train Data - Mean Absolute Error (MAE): %.6f nm\\n', meanTrainMAENm);

%% ==================== Compute & Report Modulus Metrics ==================== %%
fprintf('\\n--- Failed Modulus Curves ---\\n');
meanBadHertzCountActual = mean(arrayfun(@(x) mean((x.badHertzCountActual), 'omitnan'), results));
meanBadHertzCountPred   = mean(arrayfun(@(x) mean((x.badHertzCountPred), 'omitnan'), results));
meanBad500CountActual   = mean(arrayfun(@(x) mean((x.bad500CountActual), 'omitnan'), results));
meanBad500CountPred     = mean(arrayfun(@(x) mean((x.bad500CountPred), 'omitnan'), results));

fprintf('Mean Number of Failed Hertz Modulus Calculations (Actual, test): %.2f\\n',meanBadHertzCountActual)
fprintf('Mean Number of Failed Hertz Modulus Calculations (Predicted, test): %.2f\\n',meanBadHertzCountPred)
fprintf('Mean Number of Failed 500 nm Modulus Calculations (Actual, test): %.2f\\n',meanBad500CountActual)
fprintf('Mean Number of Failed 500 nm Modulus Calculations (Predicted, test): %.2f\\n',meanBad500CountPred)

fprintf('\\n--- Hertzian Modulus (kPa) ---\\n');
hertzMSE = mean(arrayfun(@(x) mean((x.HertzianModulusPredicted_test - x.HertzianModulusActual_test).^2, 'omitnan'), results));
hertzMAE = mean(arrayfun(@(x) mean(abs(x.HertzianModulusPredicted_test - x.HertzianModulusActual_test), 'omitnan'), results));
hertzMAPE = mean(arrayfun(@(x) mean(abs((x.HertzianModulusPredicted_test - x.HertzianModulusActual_test) ./ x.HertzianModulusActual_test)*100, 'omitnan'), results));
fprintf('Mean Squared Error (MSE): %.6f kPa^2\\n', hertzMSE);
fprintf('Mean Absolute Error (MAE): %.6f kPa\\n', hertzMAE);
fprintf('Mean Absolute Percent Error (MAPE): %.2f%%\\n', hertzMAPE);

fprintf('\\n--- Modulus at 500 nm (kPa) ---\\n');
mod500nmMSE = mean(arrayfun(@(x) mean((x.Modulus500nmPredicted_test - x.Modulus500nmActual_test).^2, 'omitnan'), results));
mod500nmMAE = mean(arrayfun(@(x) mean(abs(x.Modulus500nmPredicted_test - x.Modulus500nmActual_test), 'omitnan'), results));
mod500nmMAPE = mean(arrayfun(@(x) mean(abs((x.Modulus500nmPredicted_test - x.Modulus500nmActual_test) ./ x.Modulus500nmActual_test)*100, 'omitnan'), results));
fprintf('Mean Squared Error (MSE): %.6f kPa^2\\n', mod500nmMSE);
fprintf('Mean Absolute Error (MAE): %.6f kPa\\n', mod500nmMAE);
fprintf('Mean Absolute Percent Error (MAPE): %.2f%%\\n', mod500nmMAPE);

fprintf('\\n=== Evaluation Completed Successfully ===\\n');
fprintf('All figures are saved in the \"Evaluation_Figures\" directory.\\n');
fprintf('Bad predictions with errors exceeding %.2f kPa are logged in \"%s\".\\n', errorThreshold_kPa, badPredictionsFile);

%% ==================== Save the Collected Results to a .mat File ==================== %%
fprintf('Saving cross-validation results to \"%s\"...\\n', saveName);
save(saveName, ...
     'results', ...
     'mseScores', 'maeScores', ...
     'trainMseScores', 'trainMaeScores', ...
     'mseScoresNm', 'maeScoresNm', ...
     'trainMseScoresNm', 'trainMaeScoresNm', ...
     'hertzMSE', 'hertzMAE', 'hertzMAPE', ...
     'mod500nmMSE', 'mod500nmMAE', 'mod500nmMAPE', ...
     'YTrain_all', 'YPredTrain_all', ...
     'YTest_all', 'YPredTest_all', ...
     'meanMSE','meanMAE', ...
     'meanTrainMSE', 'meanTrainMAE', ...
     'meanMSENm', 'meanMAENm', ...
     'meanTrainMSENm', 'meanTrainMAENm',...
     'meanBad500CountPred','meanBad500CountActual',...
     'meanBadHertzCountPred','meanBadHertzCountActual');
fprintf('Cross-validation results saved successfully.\\n');

%% ==================== Plot Predicted vs Actual Values for All Folds ==================== %%
fprintf('Plotting predicted vs actual values for all folds...\\n');
plotPredictions_CNN(results, k);
fprintf('Plotting completed.\\n');

end
