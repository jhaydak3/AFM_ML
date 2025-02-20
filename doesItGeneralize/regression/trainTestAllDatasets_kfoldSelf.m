%% SCRIPT: trainTestAllDatasets_kfoldSelf.m
%
% - Trains exactly once per dataset for cross-dataset usage ("modelForOthers").
% - When we test "modelForOthers" on a different dataset, we just load the entire
%   target dataset and measure errors.
% - When training == testing the same dataset, we do *not* use leftover or "modelForSelf".
%   Instead, we perform K-fold crossvalidation (k=5 by default) on the entire same dataset.
%   The final (trainIdx,trainIdx) result is the average across folds.
%


clear; clc; close all;

%% 0) Setup
k = 5;  % number of folds for self-test crossvalidation
dataSets = [
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_LM24.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_MCF7.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_MCF10a.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_HEPG4.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_podocytes.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_tubules.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_iPSC_VSMC.mat"
];

dataSetNames = ["LM24","MCF7","MCF10a","HEPG4","Podocytes","Tubules","iPSC VSMC"];


numDatasets  = numel(dataSets);

addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions");

% Number of samples to train from each dataset for cross-dataset model
numSamplesToTrain = 10000;

% Augment the training data?
enableAugmentation = false; 

% Perform modulus metrics only on good curves? (or all curves?)
testOnlyOnGood = false;


% Define your CNN architecture
nFeatures       = 6;
sequenceLength  = 2000;
%layers = CNN_custom_pooling_after_lstm_relu(nFeatures, sequenceLength, 7);
%layers = CNN_custom2(nFeatures, sequenceLength, 7);
layers = CNN_custom_pooling_after_bilstm_2conv_relu(nFeatures, sequenceLength, 7);

% Folder to store results
resultsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\doesItGeneralize\regression\resultsKfold";
if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

% Prepare metric containers
maeNormalized = nan(numDatasets, numDatasets);
maeNm         = nan(numDatasets, numDatasets);
hertzMAPE     = nan(numDatasets, numDatasets);
mod500MAPE    = nan(numDatasets, numDatasets);

% We'll store for each dataset a struct with "modelForOthers"
trainedStructs = cell(numDatasets,1);

%% 1) Train each dataset once for cross-dataset usage
for trainIdx = 1:numDatasets
    fprintf('\n=== TRAINING CROSS-DATASET MODEL on "%s" (%s) ===\n', ...
        dataSetNames(trainIdx), dataSets(trainIdx));

    modelSaveName = fullfile(resultsFolder, ...
        sprintf('trainedInfo_%s.mat', dataSetNames(trainIdx)));

    % trainOneDataset => returns struct with "modelForOthers" only
    trainedStruct = trainOneDataset_kfoldSelf(layers, dataSets(trainIdx), ...
        numSamplesToTrain, modelSaveName, enableAugmentation);

    trainedStructs{trainIdx} = trainedStruct;
end

%% 2) Test each trained model on each dataset
for trainIdx = 1:numDatasets
    tInfo = trainedStructs{trainIdx};
    if isempty(tInfo) || isempty(tInfo.modelForOthers)
        warning('No cross-dataset model for train dataset %s => skip all tests.', dataSetNames(trainIdx));
        continue;
    end

    for testIdx = 1:numDatasets
        fprintf('\n--- TESTING model from "%s" on "%s" ---\n', ...
            dataSetNames(trainIdx), dataSetNames(testIdx));

        if trainIdx == testIdx
            % *** Perform K-fold crossvalidation on the same dataset ***
            fprintf('(Same dataset => running %d-fold crossvalidation on entire data...) \n', k);
            metricsOut = testKfoldSameDataset(layers, dataSets(testIdx), k, enableAugmentation, testOnlyOnGood);
        else
            % *** Cross dataset => use modelForOthers on entire dataset ***
            fprintf('(Cross dataset => using pre-trained modelForOthers on entire dataset) \n');
            if isempty(tInfo.modelForOthers)
                warning('No modelForOthers found => skip test on cross dataset.');
                metricsOut = makeEmptyMetrics();
            else
                metricsOut = testTrainedOnWhole(tInfo.modelForOthers, dataSets(testIdx), testOnlyOnGood);
            end
        end

        % Store results
        maeNormalized(trainIdx,testIdx) = metricsOut.maeTestNorm;
        maeNm(trainIdx,testIdx)         = metricsOut.maeTestNm;
        hertzMAPE(trainIdx,testIdx)     = metricsOut.hertzMAPE;
        mod500MAPE(trainIdx,testIdx)    = metricsOut.mod500MAPE;
    end
 end

save('bilstm_generalize_Results.mat','maeNormalized','maeNm', 'hertzMAPE','mod500MAPE', "dataSetNames")
%% 3) Generate Heatmaps
close all

fontSize = 18;
fontSizeLegend = 13;
fontFamily = 'Arial';
cellLabelColorStr = 'auto';


newIdx = [1 3 4 2 6 5 7];
maeNm2 = maeNm(newIdx, newIdx);
hertzMAPE2 = hertzMAPE(newIdx,newIdx);
dataSetNames2 = dataSetNames(newIdx);

titles = {
    "Mean Absolute Error (Normalized Units)"
    "Contact Point MAE (nm)"
    "Hertzian Modulus MAPE (%)"
    "500 nm Modulus MAPE (%)"
};

% Create rounded versions of the matrices
heatmapDataRounded = {
    round(maeNormalized, 0), 
    round(maeNm2, 0), 
    round(hertzMAPE2, 0), 
    round(mod500MAPE, 0)
};

for i = [2 3]
    figure('Name',titles{i}, ...
        'Units','inches', 'Position',[1 1 5 5]*1.3);
    h = heatmap(dataSetNames2, dataSetNames2, heatmapDataRounded{i}, ...
        'Colormap', magma, 'CellLabelColor',cellLabelColorStr,'FontSize', fontSize, ...
        'FontName',fontFamily);
    h.Position = [0.26 0.36 0.60 0.60];
    set(struct(h).NodeChildren(3), 'YTickLabelRotation', 45);
    set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45);
    xlabel('Testing Dataset');
    ylabel('Training Dataset');
    
    if i == 2
        h.ColorLimits = [min(heatmapDataRounded{i}, [], 'all'), 500];
    elseif i == 3
        h.ColorLimits = [min(heatmapDataRounded{i}, [], 'all'), 100];
    end
end

fprintf('All training/testing done. Heatmaps generated with values rounded to one decimal place.\n');








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SUBFUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function metricsOut = makeEmptyMetrics()
    metricsOut = struct('maeTestNorm',NaN,'maeTestNm',NaN,...
        'hertzMAPE',NaN,'mod500MAPE',NaN);
end


function metricsOut = testTrainedOnWhole(trainedNet, dataFile, testOnlyOnGood)
% TESTTRAINEDONWHOLE
%  Load the entire test dataset and compute metrics, with the key detail
%  that modulus calculations only use "good" curves. 

    indentationDepth_nm = 500;

    % Load data
    dataTest = load(dataFile, 'X','Y','goodOrBad', ...
        'minExtValues','maxExtValues','rawExt','rawDefl','spring_constant',...
        'R','v','th','b','fileRow','fileCol','fileIndices');
    if ~isfield(dataTest,'X') || ~isfield(dataTest,'Y')
        warning('Missing X or Y => skipping => returning NaNs');
        metricsOut = makeEmptyMetrics();
        return;
    end

    Xall = dataTest.X;
    Yall = dataTest.Y;
    N    = size(Xall,3);
    if N<1
        warning('Empty dataset => returning NaNs');
        metricsOut = makeEmptyMetrics();
        return;
    end

    % Evaluate entire set
    leftoverInds = 1:N;
    metricsOut = testTrainedModelOnDataset_sub( ...
        trainedNet, Xall, Yall, dataTest, leftoverInds, indentationDepth_nm, testOnlyOnGood);
end


function trainedStruct = trainOneDataset_kfoldSelf(layers, dataFile, numSamplesToTrain, saveModelFile, enableAugmentation)
% TRAINONEDATASET_KFOLDSelf
%   Returns a struct with .modelForOthers only (no leftover, no modelForSelf).
%   If #samples >= numSamplesToTrain => pick exactly that many => train
%   Else => use all data for training.
%
%   The "self-test" will be done later via kfold. So leftover is not needed.

    fprintf('Loading dataset "%s"...\n', dataFile);
    S = load(dataFile, 'X','Y','goodOrBad');
    if ~isfield(S,'X') || ~isfield(S,'Y')
        warning('No X or Y => returning empty struct.');
        trainedStruct = struct('modelForOthers',[]);
        return;
    end

    Xfull = S.X;
    Yfull = S.Y;
    N = size(Xfull,3);
    fprintf('  # curves = %d.\n', N);

    rng(1337,'twister');

    if N >= numSamplesToTrain
        % pick exactly numSamplesToTrain
        permAll = randperm(N);
        trainInd   = permAll(1:numSamplesToTrain);
        leftover   = permAll(numSamplesToTrain+1:end);
        fprintf('  Using %d for "modelForOthers", leftover = %d (ignored)\n', ...
            numSamplesToTrain, length(leftover));

        Xtrain = Xfull(:,:, trainInd);
        Ytrain = Yfull(trainInd);
    else
        % use ALL data
        fprintf('  < %d => using all %d for modelForOthers.\n', numSamplesToTrain, N);
        Xtrain = Xfull;
        Ytrain = Yfull;
    end

    % Optionally augment
    if enableAugmentation
        pad = 100;
        [Xaug, Yaug] = augmentData(Xtrain, Ytrain, pad);
    else
        Xaug = Xtrain;
        Yaug = Ytrain;
    end

    % Train
    netO = trainModelCore(layers, Xaug, Yaug);

    trainedStruct = struct('modelForOthers', netO);

    if ~isempty(saveModelFile)
        fprintf('Saving trained model struct to "%s"...\n', saveModelFile);
        save(saveModelFile,'trainedStruct');
    end
end


function metricsMean = testKfoldSameDataset(layers, dataFile, k, enableAugmentation, testOnlyOnGood)
% TESTKFOLDSAMEDATASET
%   Loads the entire dataset => performs k-fold crossvalidation => trains
%   a net for each fold => tests => collects the same regression metrics (MAE, MAPE).
%   Only "good" test curves are used for modulus calculation.
%   Finally returns the average metrics across folds.

    D = load(dataFile, 'X','Y','goodOrBad',...
        'minExtValues','maxExtValues','rawExt','rawDefl',...
        'spring_constant','R','v','th','b','fileRow','fileCol','fileIndices');
    if ~isfield(D,'X') || ~isfield(D,'Y')
        warning('Dataset missing X or Y => return NaN metrics.');
        metricsMean = makeEmptyMetrics();
        return;
    end

    Xfull = D.X;
    Yfull = D.Y;
    N = size(Xfull,3);

    if N < k
        warning('Not enough data to do %d-fold => return NaNs.', k);
        metricsMean = makeEmptyMetrics();
        return;
    end

    % Prepare folds
    indices = crossvalind('Kfold', N, k);

    % Storage for each fold's metrics
    maeNormAll  = zeros(k,1);
    maeNmAll    = zeros(k,1);
    hertzAll    = zeros(k,1);
    mod500All   = zeros(k,1);

    for foldID = 1:k
        fprintf('  Kfold %d of %d\n', foldID, k);

        testMask  = (indices == foldID);
        trainMask = ~testMask;

        % Split
        Xtrain = Xfull(:,:, trainMask);
        Ytrain = Yfull(trainMask);

        Xtest  = Xfull(:,:, testMask);
        Ytest  = Yfull(testMask);
        testInds= find(testMask);

        % Train
        if enableAugmentation
            [Xaug, Yaug] = augmentData(Xtrain, Ytrain, 100);
        else
            Xaug = Xtrain;
            Yaug = Ytrain;
        end
        netFold = trainModelCore(layers, Xaug, Yaug);

        % If training failed => store NaNs
        if isempty(netFold)
            maeNormAll(foldID)=NaN; maeNmAll(foldID)=NaN;
            hertzAll(foldID)=NaN;  mod500All(foldID)=NaN;
            continue;
        end

        % Evaluate on fold's test => filter good curves inside sub
        metricsFold = testTrainedModelOnDataset_sub(netFold, Xtest, Ytest, D, testInds, 500, testOnlyOnGood);

        maeNormAll(foldID) = metricsFold.maeTestNorm;
        maeNmAll(foldID)   = metricsFold.maeTestNm;
        hertzAll(foldID)   = metricsFold.hertzMAPE;
        mod500All(foldID)  = metricsFold.mod500MAPE;
    end

    % Average across folds
    metricsMean = struct();
    metricsMean.maeTestNorm = mean(maeNormAll,'omitnan');
    metricsMean.maeTestNm   = mean(maeNmAll,'omitnan');
    metricsMean.hertzMAPE   = mean(hertzAll,'omitnan');
    metricsMean.mod500MAPE  = mean(mod500All,'omitnan');
end


function metricsOut = testTrainedModelOnDataset_sub(trainedNet, Xtest, Ytest, dataStruct, testInds, indentationDepth_nm, testOnlyOnGood)
% TESTTRAINEDMODELONDATASET_SUB
%   Common subfunction to compute CP error + modulus error.
%   The modulus is only computed on the test set's "good" curves
%   (i.e. dataStruct.goodOrBad==1 for those indices).
%   The rest are ignored for modulus metrics.
%
%   This ensures that the only curves used in modulus calculations are good.
%
% Inputs:
%   - trainedNet : your dlnetwork
%   - Xtest, Ytest: a subset of the data
%   - dataStruct : a struct that has rawExt, rawDefl, goodOrBad, etc.
%   - testInds   : absolute indices in the original dataset
%   - indentationDepth_nm: for modulus calculations
%   - testOnlyOnGood: if true, only included good quality curves
%       (annotation) for the modulus metrics, if false, uses all curves
%
% Output:
%   metricsOut   : struct with:
%       .maeTestNorm
%       .maeTestNm
%       .hertzMAPE
%       .mod500MAPE
%   plus some optional fields if you want them (like MSE, etc.)

    metricsOut = struct('maeTestNorm',NaN,'maeTestNm',NaN,'hertzMAPE',NaN,'mod500MAPE',NaN);

    % Basic checks
    if isempty(trainedNet) || isempty(Xtest) || isempty(Ytest)
        return;
    end

    % 1) Predict CP in normalized units
    XTestPerm = permute(Xtest,[1,3,2]);  % [C x B x T]
    dlXTest   = dlarray(XTestPerm, 'CBT');
    YPred     = predict(trainedNet, dlXTest);
    YPred     = extractdata(YPred)';  % shape => [#samples x 1]

    testErrorsNorm = YPred - Ytest;
    maeTestNorm = mean(abs(testErrorsNorm), 'omitnan');
    metricsOut.maeTestNorm = maeTestNorm;

    % Convert to nm
    minVals = dataStruct.minExtValues(testInds)';
    maxVals = dataStruct.maxExtValues(testInds)';
    predTestNm = YPred .* (maxVals - minVals) + minVals;
    trueTestNm = Ytest .* (maxVals - minVals) + minVals;

    testErrorsNm = predTestNm - trueTestNm;
    maeTestNm = mean(abs(testErrorsNm), 'omitnan');
    metricsOut.maeTestNm = maeTestNm;

    fprintf('CP metrics: MAE(norm)=%.3f, MAE(nm)=%.3f\n', maeTestNorm, maeTestNm);

    % 2) For modulus => only "good" test curves
    if testOnlyOnGood
        goodMask = (dataStruct.goodOrBad(testInds) == 1);
        if ~any(goodMask)
            % No good curves => skip modulus
            fprintf('No good curves => skip modulus calculation.\n');
            return;
        end
    else
        goodMask = true(size(testInds));
    end

    YPredGood = YPred(goodMask);
    YTestGood = Ytest(goodMask);

    goodIndsAbs = testInds(goodMask);

    [HertzAct, HertzPred, Mod500Act, Mod500Pred] = calculateModuli( ...
        dataStruct.rawExt, dataStruct.rawDefl, ...
        YTestGood, YPredGood, ...
        goodIndsAbs, ...
        dataStruct.minExtValues, dataStruct.maxExtValues, ...
        dataStruct.b, dataStruct.th, dataStruct.R, dataStruct.v, dataStruct.spring_constant, ...
        indentationDepth_nm);

    % Filter out any NaNs
    maskH = ~isnan(HertzAct) & ~isnan(HertzPred);
    mask5 = ~isnan(Mod500Act) & ~isnan(Mod500Pred);

    if any(maskH)
        hzErrors = HertzPred(maskH) - HertzAct(maskH);
        hzAPE    = abs(hzErrors) ./ abs(HertzAct(maskH)) * 100;
        metricsOut.hertzMAPE = mean(hzAPE, 'omitnan');
    else
        metricsOut.hertzMAPE = NaN;
    end

    if any(mask5)
        m500Errors = Mod500Pred(mask5) - Mod500Act(mask5);
        m500APE    = abs(m500Errors)./abs(Mod500Act(mask5)) * 100;
        metricsOut.mod500MAPE = mean(m500APE, 'omitnan');
    else
        metricsOut.mod500MAPE = NaN;
    end

    if testOnlyOnGood
        fprintf('Modulus metrics (good test curves only): Hertz MAPE=%.2f%%, 500nm MAPE=%.2f%%\n', ...
            metricsOut.hertzMAPE, metricsOut.mod500MAPE);
    else
        fprintf('Modulus metrics: Hertz MAPE=%.2f%%, 500nm MAPE=%.2f%%\n', ...
            metricsOut.hertzMAPE, metricsOut.mod500MAPE);
    end
end


