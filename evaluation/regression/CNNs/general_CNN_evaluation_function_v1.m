function [meanMSE, meanMAE, meanTrainMSE, meanTrainMAE] = general_CNN_evaluation_function_v1(layers,preprocessedDataFile, saveName, useAllFeatures, useAugmentation)
%EVALUATE_MODEL_REGRESSION_CNN_CUSTOM_REGRESSION_V5_MODIFIED
% Performs k-fold cross-validation for regression using a custom CNN model.
% Calculates regression metrics for Contact Points (CP) and defers Modulus 
% (Hertzian and 500 nm) calculations until after the k-fold loop.
% Logs bad predictions exceeding a specified error threshold.
%
% Outputs:
%   meanMSE       - Mean Squared Error over k folds (Test Data)
%   meanMAE       - Mean Absolute Error over k folds (Test Data)
%   meanTrainMSE  - Mean Squared Error over k folds (Training Data)
%   meanTrainMAE  - Mean Absolute Error over k folds (Training Data)
%
% Saves:
%   'crossoverResults_regression_CNN_custom.mat' containing predictions and indices
%   Scatter plots in 'Evaluation_Figures' directory
%   'bad_predictions.txt' logging predictions exceeding the error threshold
% if useAllFeatures is true, uses expanded. if false, uses only raw
% deflection curve
clc;
close all;
fprintf('Starting k-fold cross-validation using Custom CNN Regression Model...\n');

%% ==================== Configuration Parameters ==================== %%
k = 5; 
pad = 100; 

% Define CNN model parameters (consistent with training)
filter1 = 7; 
filter2 = 7; 

% Error threshold for logging bad predictions (in kPa)
errorThreshold_kPa = 7.5;

% Set the seed for the random number generator
seedNumber = 1337;
rng(seedNumber);

% Define the depth for the pointwise modulus evaluation
indentationDepth_nm = 500; % nm

% If true, modulus metrics will only come from good quality curves.
testOnlyOnGood = false;



%% ==================== Load the Preprocessed Data ==================== %%
fprintf('Loading preprocessed data from "%s"...\n', preprocessedDataFile);
data = load(preprocessedDataFile);

% Validate loaded data
requiredFieldsPD = {'X', 'Y', 'maxExtValues', 'minExtValues', 'rawExt', 'rawDefl', ...
                    'fileIndices', 'fileRow', 'fileCol', ...
                    'b','th','R','v','spring_constant','goodOrBad'};
for i = 1:length(requiredFieldsPD)
    if ~isfield(data, requiredFieldsPD{i})
        error('Field "%s" is missing from "%s".', requiredFieldsPD{i}, preprocessedDataFile);
    end
end

X = data.X; % [numFeatures x sequenceLength x numSamples]
Y = data.Y; % [numSamples x 1]
maxExtValues = data.maxExtValues';
minExtValues = data.minExtValues';
goodOrBad = data.goodOrBad;

numSamples = size(X, 3);
fprintf('Loaded data with %d samples.\n', numSamples);

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
% This section is a little bit more complicated than usual. This is because
% we are training on every curve that has an identifiable contact point,
% but not all of those curves are good quality. So while metrics for how
% off you are from the contact point are meaningful, downstream metrics on
% modulus predictions aren't really meaningful, as we wouldn't include
% those in a real analysis. So here, we split up good and bad curves so
% that we can balance the folds (same number of good and bad curves)

fprintf('Setting up %d-fold cross-validation...\n', k);

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
fprintf('Starting k-fold cross-validation with %d folds...\n', k);

for fold = 1:k
    fprintf('Processing Fold %d of %d...\n', fold, k);

    % Split data into training and testing based on current fold
    testIdx = (indices == fold);
    trainIdx = ~testIdx;


    if useAllFeatures
        XTrain = X(:, :, trainIdx);       % [numFeatures x sequenceLength x numTrainSamples]
        XTest = X(:, :, testIdx);         % [numFeatures x sequenceLength x numTestSamples]

    else
        XTrain = X(1, :, trainIdx);       % [numFeatures x sequenceLength x numTrainSamples]
        XTest = X(1, :, testIdx);         % [numFeatures x sequenceLength x numTestSamples]
    end
    
    YTrain = Y(trainIdx);             % [numTrainSamples x 1]
    YTest = Y(testIdx);               % [numTestSamples x 1]

    fprintf('Fold %d: Training samples = %d | Testing samples = %d\n', fold, sum(trainIdx), sum(testIdx));

    %% ==================== Augment the Training Data ==================== %%
    if useAugmentation == true
        fprintf('Fold %d: Augmenting training data...\n', fold);
        [XTrainAug, YTrainAug] = augmentData(XTrain, YTrain, pad);
    else 
        XTrainAug = XTrain;
        YTrainAug = YTrain;
    end

    %% ==================== Define the Regression Neural Network ==================== %%
    fprintf('Fold %d: Defining the CNN architecture...\n', fold);
    %layers = CNN_custom(size(XTrainAug,1), size(XTrainAug,2), filter1, filter2);
    % This is now an input to the function.

    net = dlnetwork(layers);

    %% ==================== Prepare Training Data ==================== %%
    fprintf('Fold %d: Preparing training data...\n', fold);

    %% ==================== Train the Network ==================== %%
    fprintf('Starting training of the neural network for fold %d...\n', fold);
    predictSecondWay = false;
    try
        tic
        %[trainedNet, info] = trainnet(XTrainDL, YTrainAug, net, "mae", options);
        trainedNet = trainModelCore(layers, XTrainAug, YTrainAug);
        fprintf('Training completed successfully for fold %d.\n', fold);
        timeElapsed = toc;
    catch ME
        disp(getReport(ME, 'extended')); 
        warning('Error during training for fold %d: %s. \n Trying with trainModelCore2.\n', fold, ME.message);
        try
           % Some of the networks need to be treated as images, and these
           % need to use trainModelCore2, not trainModelCore.
            tic
            trainedNet = trainModelCore2(layers, XTrainAug, YTrainAug);
            timeElapsed = toc;
            % If this worked, set a flag to predict the second way.
            predictSecondWay = true;
        catch ME
            disp(getReport(ME, 'extended'));
            error('Error during training for fold %d: %s.\n', fold, ME.message);
        end
    end

    %% ==================== Predictions on Training & Testing Data ==================== %%

    if predictSecondWay == false
        % Training
        XTrainAugForPred = permute(XTrainAug, [1,3,2]); % [C x B x T]
        XTrainAugForPred = dlarray(XTrainAugForPred, 'CBT');
    
        fprintf('Fold %d: Making predictions on training data...\n', fold);
        YPredTrainAug = predict(trainedNet, XTrainAugForPred);  
        YPredTrainAug = extractdata(YPredTrainAug)';
        %YPredTrain = YPredTrainAug(1:length(YTrain));  % Only original (non-augmented)
        YPredTrain = YPredTrainAug;
    
        % Testing
        fprintf('Fold %d: Making predictions on testing data...\n', fold);
        XTestPermuted = permute(XTest, [1, 3, 2]);  
        XTestDL = dlarray(XTestPermuted, 'CBT');
    
        XTestForPred =  permute(XTest, [1,3,2]); % [C x B x T]
        XTestForPred = dlarray(XTestForPred , 'CBT');
    
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
        numFeatures  = size(XTrainAug,1);   % C
        sequenceLen  = size(XTrainAug,2);   % T
        batchSize    = size(XTrainAug,3);   % B
       
        XTrainAugForPred = permute(XTrainAug, [2 4 1 3]);
        XTrainAugForPred = reshape(XTrainAugForPred, [sequenceLen, 1, numFeatures, batchSize]);
        YPredTrain = predict(trainedNet,XTrainAugForPred);

        numFeaturesTest  = size(XTest,1);   % C
        sequenceLenTest  = size(XTest,2);   % T
        batchSizeTest    = size(XTest,3);   % B
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

    predTestNm = YPredTest .* (maxExtValues(testIdx) - minExtValues(testIdx)) + minExtValues(testIdx);
    trueTestNm = YTest .* (maxExtValues(testIdx) - minExtValues(testIdx)) + minExtValues(testIdx);
    testErrorsNm = predTestNm - trueTestNm;
    mseScoresNm(fold) = mean(testErrorsNm.^2);
    maeScoresNm(fold) = mean(abs(testErrorsNm));

    %% ==================== Store Predictions in Results Struct ==================== %%
    YTrain_all{fold} = YTrain;
    YTest_all{fold} = YTest;
    YPredTrain_all{fold} = YPredTrain;
    YPredTest_all{fold} = YPredTest;

    results(fold).fold = fold;
    results(fold).trainIndices = find(trainIdx);
    results(fold).testIndices = find(testIdx);
    results(fold).YTrain = YTrain;
    results(fold).YPredTrain = YPredTrain;
    results(fold).YTest = YTest;
    results(fold).YPredTest = YPredTest;
    results(fold).YTrainNm = trueTrainNm;
    results(fold).YPredTrainNm = predTrainNm;
    results(fold).YTestNm = trueTestNm;
    results(fold).YPredTestNm = predTestNm;
    results(fold).timeToTrain = timeElapsed;


    fprintf('Fold %d: Completed.\n', fold);
end

%% ==================== Calculate Moduli & Log Bad Predictions AFTER the k-fold loop ==================== %%
badPredictionsFile = 'bad_predictions.txt';
fileID = fopen(badPredictionsFile, 'w');  % Overwrite existing
if fileID == -1
    error('Cannot open file %s for writing.', badPredictionsFile);
else
    fprintf(fileID, 'Curves with Absolute Prediction Error > %.2f kPa\n', errorThreshold_kPa);
    fprintf(fileID, 'Fold\tCurveIdx\tHertzianError(kPa)\t500nmError(kPa)\tFile\tRow\tCol\n');
end

fprintf('Calculating modulus metrics and logging bad predictions...\n');
for fold = 1:k

    fprintf('Fold %d: Calculating modulus metrics...\n', fold);

    % Retrieve predictions and indices from 'results'
    YTrain = results(fold).YTrain;
    YPredTrain = results(fold).YPredTrain;
    YTest = results(fold).YTest;
    YPredTest = results(fold).YPredTest;
    trainIdx = results(fold).trainIndices;
    testIdx = results(fold).testIndices;

    if testOnlyOnGood

        % Remove low quality curves from the training and testing
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
    
    
        % Calculate Moduli for Training Data
        [HertzianModulusActualTrain, HertzianModulusPredictedTrain, ...
         Modulus500nmActualTrain, Modulus500nmPredictedTrain] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTrainEval, YPredTrainEval, ...
            trainIdxGood, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);
    
        % Calculate Moduli for Testing Data
        [HertzianModulusActualTest, HertzianModulusPredictedTest, ...
            Modulus500nmActualTest, Modulus500nmPredictedTest] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTestEval, YPredTestEval, ...
            testIdxGood, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);
    else
        % Calculate Moduli for Training Data
        [HertzianModulusActualTrain, HertzianModulusPredictedTrain, ...
            Modulus500nmActualTrain, Modulus500nmPredictedTrain] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTrain, YPredTrain, ...
            trainIdx, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);

        % Calculate Moduli for Testing Data
        [HertzianModulusActualTest, HertzianModulusPredictedTest, ...
            Modulus500nmActualTest, Modulus500nmPredictedTest] = calculateModuli( ...
            data.rawExt, data.rawDefl, YTest, YPredTest, ...
            testIdx, minExtValues, maxExtValues, ...
            data.b, data.th, data.R, data.v, data.spring_constant, ...
            indentationDepth_nm);

    end

    badHertzCountActual = sum(isnan(HertzianModulusActualTest));
    bad500CountActual = sum(isnan(Modulus500nmActualTest));

    badHertzCountPred = sum(isnan(HertzianModulusPredictedTest));
    bad500CountPred = sum(isnan(Modulus500nmPredictedTest));

    % Store them into results struct
    results(fold).HertzianModulusActual_train       = HertzianModulusActualTrain;
    results(fold).HertzianModulusPredicted_train    = HertzianModulusPredictedTrain;
    results(fold).Modulus500nmActual_train          = Modulus500nmActualTrain;
    results(fold).Modulus500nmPredicted_train       = Modulus500nmPredictedTrain;
    results(fold).HertzianModulusActual_test        = HertzianModulusActualTest;
    results(fold).HertzianModulusPredicted_test     = HertzianModulusPredictedTest;
    results(fold).Modulus500nmActual_test           = Modulus500nmActualTest;
    results(fold).Modulus500nmPredicted_test        = Modulus500nmPredictedTest;

    results(fold).badHertzCountActual               = badHertzCountActual;
    results(fold).badHertzCountPred                 = badHertzCountPred;
    results(fold).bad500CountActual                 = bad500CountActual;
    results(fold).bad500CountPred                   = bad500CountPred;





    %% =============== Log Bad Predictions for This Fold =============== %%
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

        fprintf(fileID, '%d\t%d\t%.3f\t%.3f\t%s\t%d\t%d\n', ...
            fold, curveIdx, ...
            absErrorHertz(badIndices(j)), ...
            absError500nm(badIndices(j)), ...
            fileName, row, col);
    end

end
fclose(fileID);
fprintf('Bad predictions saved to "%s".\n', badPredictionsFile);

%% ==================== Calculate Mean Performance Metrics Across All Folds ==================== %%
fprintf('Calculating mean performance metrics across all folds...\n');
validFolds = ~isnan(mseScores) & ~isnan(maeScores) & ~isnan(mseScoresNm) & ~isnan(maeScoresNm);

meanMSE = mean(mseScores(validFolds));
meanMAE = mean(maeScores(validFolds));
meanTrainMSE = mean(trainMseScores(validFolds));
meanTrainMAE = mean(trainMaeScores(validFolds));

meanMSENm = mean(mseScoresNm(validFolds));
meanMAENm = mean(maeScoresNm(validFolds));
meanTrainMSENm = mean(trainMseScoresNm(validFolds));
meanTrainMAENm = mean(trainMaeScoresNm(validFolds));

%% ==================== Report Contact Points Metrics ==================== %%
fprintf('\n=== Overall Performance Metrics Across All Folds ===\n');

% Contact Points (Normalized)
fprintf('\n--- Contact Points (Normalized) ---\n');
fprintf('Mean Squared Error (MSE): %.6f\n', meanMSE);
fprintf('Mean Absolute Error (MAE): %.6f\n', meanMAE);
fprintf('Train Data - Mean Squared Error (MSE): %.6f\n', meanTrainMSE);
fprintf('Train Data - Mean Absolute Error (MAE): %.6f\n', meanTrainMAE);

% Contact Points (nm)
fprintf('\n--- Contact Points (nm) ---\n');
fprintf('Mean Squared Error (MSE): %.6f nm^2\n', meanMSENm);
fprintf('Mean Absolute Error (MAE): %.6f nm\n', meanMAENm);
fprintf('Train Data - Mean Squared Error (MSE): %.6f nm^2\n', meanTrainMSENm);
fprintf('Train Data - Mean Absolute Error (MAE): %.6f nm\n', meanTrainMAENm);

%% ==================== Compute & Report Modulus Metrics ==================== %%
fprintf('\n--- Failed Modulus Curves ---\n');
meanBadHertzCountActual = mean(arrayfun(@(x) mean((x.badHertzCountActual), 'omitnan'), results));
meanBadHertzCountPred = mean(arrayfun(@(x) mean((x.badHertzCountPred), 'omitnan'), results));
meanBad500CountActual = mean(arrayfun(@(x) mean((x.bad500CountActual), 'omitnan'), results));
meanBad500CountPred = mean(arrayfun(@(x) mean((x.bad500CountPred), 'omitnan'), results));

fprintf('Mean Number of Failed Hertz Modulus Calculations (Actual, test): %.2f\n',meanBadHertzCountActual)
fprintf('Mean Number of Failed Hertz Modulus Calculations (Predicted, test): %.2f\n',meanBadHertzCountPred)
fprintf('Mean Number of Failed 500 nm Modulus Calculations (Actual, test): %.2f\n',meanBad500CountActual)
fprintf('Mean Number of Failed 500 nm Modulus Calculations (Predicted, test): %.2f\n',meanBad500CountPred)







fprintf('\n--- Hertzian Modulus (kPa) ---\n');
hertzMSE = mean(arrayfun(@(x) mean((x.HertzianModulusPredicted_test - x.HertzianModulusActual_test).^2, 'omitnan'), results));
hertzMAE = mean(arrayfun(@(x) mean(abs(x.HertzianModulusPredicted_test - x.HertzianModulusActual_test), 'omitnan'), results));
hertzMAPE = mean(arrayfun(@(x) mean(abs((x.HertzianModulusPredicted_test - x.HertzianModulusActual_test) ./ x.HertzianModulusActual_test)*100, 'omitnan'), results));
fprintf('Mean Squared Error (MSE): %.6f kPa^2\n', hertzMSE);
fprintf('Mean Absolute Error (MAE): %.6f kPa\n', hertzMAE);
fprintf('Mean Absolute Percent Error (MAPE): %.2f%%\n', hertzMAPE);

fprintf('\n--- Modulus at 500 nm (kPa) ---\n');
mod500nmMSE = mean(arrayfun(@(x) mean((x.Modulus500nmPredicted_test - x.Modulus500nmActual_test).^2, 'omitnan'), results));
mod500nmMAE = mean(arrayfun(@(x) mean(abs(x.Modulus500nmPredicted_test - x.Modulus500nmActual_test), 'omitnan'), results));
mod500nmMAPE = mean(arrayfun(@(x) mean(abs((x.Modulus500nmPredicted_test - x.Modulus500nmActual_test) ./ x.Modulus500nmActual_test)*100, 'omitnan'), results));
fprintf('Mean Squared Error (MSE): %.6f kPa^2\n', mod500nmMSE);
fprintf('Mean Absolute Error (MAE): %.6f kPa\n', mod500nmMAE);
fprintf('Mean Absolute Percent Error (MAPE): %.2f%%\n', mod500nmMAPE);

fprintf('\n=== Evaluation Completed Successfully ===\n');
fprintf('All figures are saved in the "Evaluation_Figures" directory.\n');
fprintf('Bad predictions with errors exceeding %.2f kPa are logged in "%s".\n', errorThreshold_kPa, badPredictionsFile);

%% ==================== Save the Collected Results to a .mat File ==================== %%
fprintf('Saving cross-validation results to "%s"...\n', saveName);
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
     'meanTrainMSENm', 'meanTrainMAENm',"meanBad500CountPred","meanBad500CountActual", ...
     "meanBadHertzCountPred","meanBadHertzCountActual");
fprintf('Cross-validation results saved successfully.\n');

%% ==================== Plot Predicted vs Actual Values for All Folds ==================== %%
fprintf('Plotting predicted vs actual values for all folds...\n');
plotPredictions_CNN(results, k);
fprintf('Plotting completed.\n');

end

