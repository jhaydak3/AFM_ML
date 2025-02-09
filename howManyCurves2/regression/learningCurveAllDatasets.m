%% learningCurveAllDatasets_regressionHoldout.m
%
% This script performs a learning-curve style analysis for *regression* data,
% but uses a fixed holdout portion (size=holdoutSize) for testing, never used in training.
%
% Steps:
%   1) For each dataset, randomly pick 'holdoutSize' curves => "universal" test set.
%   2) For each train-size sVal, do R repeated draws of sVal from the leftover portion => train => evaluate on holdout.
%   3) Collate mean ± std of metrics (MAE, MAPE, etc.) across R runs => learning curve.
%
% By: <Your Name>
% Date: <Date>

clc; clear; close all;

%% 0) Setup

% Example: regression data sets
dataSets = [
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat"
    % Add more file paths as needed, one per dataset
];
dataSetNames = ["All"]; % Provide matching names
numDatasets  = numel(dataSets);

% Train sizes to explore
trainSizes = [100:50:750, 1000:500:4000];

% Number of repeated random draws at each train size
R = 5;

% Fixed holdout size for testing
holdoutSize = 1000;
fprintf('Using a fixed holdout of %d samples for each dataset.\n', holdoutSize);

% CNN architecture (example)
nFeatures      = 6;
sequenceLength = 2000;
layers = CNN_custom_pooling_after_lstm_2conv_relu(nFeatures, sequenceLength, 7);

% Data augmentation?
enableAugmentation = false;

% Depth for single-point modulus calculation (500 nm typical)
indentationDepth_nm = 500;

% Initialize results struct
allResults = struct();

rng(1337);  % for reproducibility

%% 1) Loop over each dataset
for d = 1:numDatasets
    fprintf('\n=== LEARNING CURVE (Holdout) for dataset: %s ===\n', dataSetNames(d));

    dataFile = dataSets(d);
    dataTrain = load(dataFile, ...
        'X','Y','goodOrBad','minExtValues','maxExtValues','rawExt','rawDefl',...
        'spring_constant','R','v','th','b','fileRow','fileCol','fileIndices');
    if ~isfield(dataTrain,'X') || ~isfield(dataTrain,'Y')
        warning('File "%s" missing X or Y => skip.\n', dataFile);
        continue;
    end

    Xfull = dataTrain.X;  % [features x seqLen x N]
    Yfull = dataTrain.Y;  % [N x 1] (regression target)
    N     = size(Xfull,3);

    fprintf('  Found %d samples in %s.\n', N, dataFile);
    if N < 2
        warning('Not enough data => skip.\n');
        continue;
    end

    % (a) Create universal holdout of 'holdoutSize' (or adjust if too big)
    if holdoutSize >= N
        holdoutSize = N - 1;
        fprintf('WARNING: holdoutSize >= N => adjusting holdout to %d\n', holdoutSize);
    end

    permAll   = randperm(N);
    holdInds  = permAll(1:holdoutSize);
    leftoverInds = permAll(holdoutSize+1:end);
    leftoverN = numel(leftoverInds);

    fprintf('  Creating holdout of %d. Remainder for training: %d\n', holdoutSize, leftoverN);

    % Filter trainSizes => must not exceed leftoverN
    validSizes = trainSizes(trainSizes <= leftoverN);
    if isempty(validSizes)
        warning('No valid train sizes for dataset "%s" => skipping.\n', dataSetNames(d));
        continue;
    end

    % Prepare arrays for metrics => [#validSizes x R]
    maeNormMatrix    = nan(numel(validSizes), R);
    maeNmMatrix      = nan(numel(validSizes), R);
    hertzMapeMatrix  = nan(numel(validSizes), R);
    mod500MapeMatrix = nan(numel(validSizes), R);

    %% 2) For each train size
    for sIdx = 1:numel(validSizes)
        sVal = validSizes(sIdx);
        fprintf('  Train size = %d\n', sVal);

        for rDraw = 1:R
            % (i) Random subset from leftoverInds for training
            rp = randperm(leftoverN);
            pickTrain = leftoverInds(rp(1:sVal));

            % Build training set
            XtrainRaw = Xfull(:,:, pickTrain);
            YtrainRaw = Yfull(pickTrain);

            % Augment (optional)
            if enableAugmentation
                [Xaug, Yaug] = augmentData(XtrainRaw, YtrainRaw, 100);
                XtrainAll = Xaug;
                YtrainAll = Yaug;
                fprintf('    [Augment] Raw=%d => total after augment=%d\n', ...
                    sVal, size(XtrainAll,3));
            else
                XtrainAll = XtrainRaw;
                YtrainAll = YtrainRaw;
            end

            % Train
            trainedNet = trainModelCore(layers, XtrainAll, YtrainAll);
            if isempty(trainedNet)
                maeNormMatrix(sIdx, rDraw)    = NaN;
                maeNmMatrix(sIdx, rDraw)      = NaN;
                hertzMapeMatrix(sIdx, rDraw)  = NaN;
                mod500MapeMatrix(sIdx, rDraw) = NaN;
                continue;
            end

            % Evaluate on the universal holdout
            Xhold = Xfull(:,:, holdInds);
            Yhold = Yfull(holdInds);

            metricsOut = testTrainedModelOnDataset_sub( ...
                trainedNet, Xhold, Yhold, dataTrain, holdInds, indentationDepth_nm);

            maeNormMatrix(sIdx, rDraw)    = metricsOut.maeTestNorm;
            maeNmMatrix(sIdx, rDraw)      = metricsOut.maeTestNm;
            hertzMapeMatrix(sIdx, rDraw)  = metricsOut.hertzMAPE;
            mod500MapeMatrix(sIdx, rDraw) = metricsOut.mod500MAPE;
        end
    end

    %% 3) Compute mean & std across R draws
    meanMaeNorm   = mean(maeNormMatrix, 2, 'omitnan');
    stdMaeNorm    = std(maeNormMatrix, 0, 2, 'omitnan');
    meanMaeNm     = mean(maeNmMatrix,   2, 'omitnan');
    stdMaeNm      = std(maeNmMatrix,    0, 2, 'omitnan');
    meanHertzMape = mean(hertzMapeMatrix,  2, 'omitnan');
    stdHertzMape  = std(hertzMapeMatrix,   0, 2, 'omitnan');
    mean500Mape   = mean(mod500MapeMatrix, 2, 'omitnan');
    std500Mape    = std(mod500MapeMatrix,  0, 2, 'omitnan');

    %% 4) Store results
    allResults(d).cellName      = dataSetNames(d);
    allResults(d).N             = N;
    allResults(d).holdoutSize   = holdoutSize;
    allResults(d).trainSizes    = validSizes;

    allResults(d).maeNormMatrix    = maeNormMatrix;
    allResults(d).maeNmMatrix      = maeNmMatrix;
    allResults(d).hertzMapeMatrix  = hertzMapeMatrix;
    allResults(d).mod500MapeMatrix = mod500MapeMatrix;

    allResults(d).meanMaeNorm   = meanMaeNorm;
    allResults(d).stdMaeNorm    = stdMaeNorm;
    allResults(d).meanMaeNm     = meanMaeNm;
    allResults(d).stdMaeNm      = stdMaeNm;
    allResults(d).meanHertzMape = meanHertzMape;
    allResults(d).stdHertzMape  = stdHertzMape;
    allResults(d).mean500Mape   = mean500Mape;
    allResults(d).std500Mape    = std500Mape;
end

%% 5) Plot results
figure('Name','CP Errors (Normalized, nm) vs Train Size (Holdout)');
tiledlayout('vertical','TileSpacing','compact');
ax1 = nexttile(); hold on; grid on; title('MAE (Normalized)');
ax2 = nexttile(); hold on; grid on; title('MAE (nm)');

colors = lines(numDatasets);

for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end
    sVals   = allResults(d).trainSizes;
    mMaeN   = allResults(d).meanMaeNorm;
    eMaeN   = allResults(d).stdMaeNorm;
    mMaeNm  = allResults(d).meanMaeNm;
    eMaeNm  = allResults(d).stdMaeNm;

    axes(ax1);
    errorbar(sVals, mMaeN, eMaeN, '-o','Color',colors(d,:), ...
        'DisplayName', allResults(d).cellName);

    axes(ax2);
    errorbar(sVals, mMaeNm, eMaeNm, '-o','Color',colors(d,:), ...
        'DisplayName', allResults(d).cellName);
end
legend(ax1,'Location','best');
legend(ax2,'Location','best');
xlabel(ax1,'Training set size'); ylabel(ax1,'MAE (norm)');
xlabel(ax2,'Training set size'); ylabel(ax2,'MAE (nm)');

figure('Name','Modulus MAPE (Hertz, 500nm) vs Train Size (Holdout)');
tiledlayout('vertical','TileSpacing','compact');
ax3 = nexttile(); hold on; grid on; title('Hertz MAPE (%)');
ax4 = nexttile(); hold on; grid on; title('500 nm MAPE (%)');

for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end
    sVals   = allResults(d).trainSizes;
    mHertz  = allResults(d).meanHertzMape;
    eHertz  = allResults(d).stdHertzMape;
    m500    = allResults(d).mean500Mape;
    e500    = allResults(d).std500Mape;

    axes(ax3);
    errorbar(sVals, mHertz, eHertz, '-o','Color',colors(d,:), ...
        'DisplayName', allResults(d).cellName);

    axes(ax4);
    errorbar(sVals, m500, e500, '-o','Color',colors(d,:), ...
        'DisplayName', allResults(d).cellName);
end
legend(ax3,'Location','best');
legend(ax4,'Location','best');
xlabel(ax3,'Training set size'); ylabel(ax3,'MAPE (%)');
xlabel(ax4,'Training set size'); ylabel(ax4,'MAPE (%)');

%% Additional Plots
% 1) Single plot: Contact point (nm) vs Training Set Size
figure('Name','Learning Curve - Contact Point (nm)');
hold on; grid on;
colors = lines(numDatasets);

for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end

    sVals  = allResults(d).trainSizes;
    mMaeNm = allResults(d).meanMaeNm;
    eMaeNm = allResults(d).stdMaeNm;

    errorbar(sVals, mMaeNm, eMaeNm, '-o', ...
        'Color', 'k', ...
        'DisplayName', allResults(d).cellName);
end

xlabel('Training Set Size');
ylabel('Contact Point MAE (nm)');
title('Learning Curve - Contact Point MAE (Holdout = 1000)');
legend('Location','best');

% 2) Single plot: Hertz MAPE (%) vs Training Set Size
figure('Name','Learning Curve - Hertzian Modulus MAPE (Holdout = 1000)');
hold on; grid on;

for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end

    sVals  = allResults(d).trainSizes;
    mHertz = allResults(d).meanHertzMape;
    eHertz = allResults(d).stdHertzMape;

    errorbar(sVals, mHertz, eHertz, '-o', ...
        'Color', 'k', ...
        'DisplayName', allResults(d).cellName);
end

xlabel('Training Set Size');
ylabel('Hertzian Modulus MAPE (%)');
title('Learning Curve - Hertzian Modulus MAPE (Holdout = 1000)');
legend('Location','best');


%% 6) Save final results
saveFile = sprintf('learningCurveResults_regressionHoldout_%d.mat', holdoutSize);
save(saveFile,'allResults','numDatasets','trainSizes','holdoutSize');
fprintf('\nAll done! Results saved in %s.\n', saveFile);




%% ------------------------------------------------------------------------


%% ------------------------------------------------------------------------
%% HELPER: yourMAPEcalculations
%% ------------------------------------------------------------------------
function [hertzMAPE, mod500MAPE] = yourMAPEcalculations(pred, Ytest, dataStruct, testInds, indentationDepth_nm)
% Placeholder. Replace with your actual code that calculates:
%  - MAPE for Hertz model
%  - MAPE for 500 nm modulus
% etc.

hertzMAPE  = rand() * 5;   % dummy example
mod500MAPE = rand() * 10;  % dummy example
end
