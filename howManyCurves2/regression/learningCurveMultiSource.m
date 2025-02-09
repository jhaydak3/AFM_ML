% LEARNINGCURVEMULTISOURCE_HOLDOUT
%
% Similar to the multi-source learning curve script, but uses a single,
% universal holdout from EACH dataset, never used in training.
%
% Steps:
% 1) Load all 7 datasets in the order:
%       1. tubules
%       2. podocytes
%       3. HEPG4
%       4. iPSC
%       5. LM24
%       6. MCF7
%       7. MCF10a
% 2) For each dataset, randomly pick 'holdoutSize' curves => store them in
%    a universal holdout struct that has all 7 sets' holdouts combined.
%    Remove these from the dataset so they can never be used for training.
% 3) For i = 1..7:
%     - Combine leftover from the first i sets as a training pool (the rest
%       are ignored).
%     - For each trainSize, do repeated draws => train => test on the same
%       single universal holdout => record metrics.
% 4) Plot MAE (norm, nm), MAPE (hertz, 500nm) across train sizes, for i=1..7.
% 5) Save results.

clear; clc; close all;

%% 0) Set up data paths in the EXACT order you requested:
dataPaths = [
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_tubules.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_podocytes.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_HEPG4.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_iPSC_VSMC.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_LM24.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_MCF7.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_MCF10a.mat"
    ];

cellTypeNames = [ ...
    "Tubules","Podocytes","HEPG4","iPSC","LM24","MCF7","MCF10a" ...
    ];
numAllDatasets = numel(dataPaths);

%% (a) You can change holdoutSize if you want
holdoutSize = 100;   % # curves per dataset for the universal holdout

% Example training sizes
trainSizes = [100:100:1000, 1500, 2000, 2500, 3000];  
R = 5;  % # repeated random draws

enableAugment = false;   % set true if you want data augmentation
indentationDepth_nm = 500;  % typically 500 nm

%% (b) Architecture: pick your regression network
% example: "CNN_custom_pooling_after_lstm_2conv_relu"
nFeatures = 6;
seqLength = 2000;
layers = CNN_custom_pooling_after_lstm_2conv_relu(nFeatures, seqLength, 7);

%% 1) Load all data into a struct array
allData = struct();
for idx = 1:numAllDatasets
    S = load(dataPaths(idx), 'X','Y','goodOrBad','minExtValues','maxExtValues', ...
        'rawExt','rawDefl','spring_constant','R','v','th','b','fileRow','fileCol','fileIndices');

    allData(idx).X = S.X;   % [features x 2000 x N]
    allData(idx).Y = S.Y;   % [N x 1]
    allData(idx).N = size(S.X,3);
    allData(idx).name = cellTypeNames(idx);

    % We'll store the other relevant fields for potential merging:
    allData(idx).minExtValues = S.minExtValues;
    allData(idx).maxExtValues = S.maxExtValues;
    allData(idx).rawExt       = S.rawExt;
    allData(idx).rawDefl      = S.rawDefl;
    allData(idx).goodOrBad    = S.goodOrBad;
    allData(idx).b            = S.b;
    allData(idx).th           = S.th;
    allData(idx).R            = S.R;
    allData(idx).v            = S.v;
    allData(idx).spring_constant = S.spring_constant;

    fprintf('Loaded dataset "%s": %d curves.\n', cellTypeNames(idx), allData(idx).N);
end

%% 2) Create the universal holdout of 'holdoutSize' from each dataset
holdoutStruct = initEmptyDataStruct();
Xhold = [];
Yhold = [];

for idx = 1:numAllDatasets
    dsN = allData(idx).N;

    if holdoutSize > dsN
        warning('Requested holdoutSize=%d > total curves %d in dataset %s', ...
            holdoutSize, dsN, allData(idx).name);
        hSizeReal = dsN;  % fallback: hold out all
    else
        hSizeReal = holdoutSize;
    end

    permIdx = randperm(dsN);
    holdInds = permIdx(1:hSizeReal);
    remainInds= permIdx(hSizeReal+1:end);

    % Build holdout subset from this dataset
    Xh = allData(idx).X(:,:, holdInds);
    Yh = allData(idx).Y(holdInds);

    % Add them to Xhold, Yhold
    Xhold = cat(3, Xhold, Xh);
    Yhold = cat(1, Yhold, Yh);

    % Fill holdoutStruct fields
    holdoutStruct.minExtValues  = [holdoutStruct.minExtValues, allData(idx).minExtValues(holdInds)];
    holdoutStruct.maxExtValues  = [holdoutStruct.maxExtValues, allData(idx).maxExtValues(holdInds)];
    holdoutStruct.rawExt        = [holdoutStruct.rawExt,       allData(idx).rawExt(holdInds)];
    holdoutStruct.rawDefl       = [holdoutStruct.rawDefl,      allData(idx).rawDefl(holdInds)];
    holdoutStruct.goodOrBad     = [holdoutStruct.goodOrBad,    allData(idx).goodOrBad(holdInds)];
    holdoutStruct.b             = [holdoutStruct.b,            allData(idx).b(holdInds)];
    holdoutStruct.th            = [holdoutStruct.th,           allData(idx).th(holdInds)];
    holdoutStruct.R             = [holdoutStruct.R,            allData(idx).R(holdInds)];
    holdoutStruct.v             = [holdoutStruct.v,            allData(idx).v(holdInds)];
    holdoutStruct.spring_constant = [holdoutStruct.spring_constant, allData(idx).spring_constant(holdInds)];

    % Remove these holdout from the training portion
    allData(idx).X(:,:,holdInds) = [];
    allData(idx).Y(holdInds)     = [];
    allData(idx).minExtValues(holdInds) = [];
    allData(idx).maxExtValues(holdInds) = [];
    allData(idx).rawExt(holdInds)       = [];
    allData(idx).rawDefl(holdInds)      = [];
    allData(idx).goodOrBad(holdInds)    = [];
    allData(idx).b(holdInds)            = [];
    allData(idx).th(holdInds)           = [];
    allData(idx).R(holdInds)            = [];
    allData(idx).v(holdInds)            = [];
    allData(idx).spring_constant(holdInds) = [];

    allData(idx).N = dsN - hSizeReal;  % leftover for training
    fprintf('  Dataset %s: held out %d, leftover for train=%d.\n', ...
        allData(idx).name, hSizeReal, allData(idx).N);
end

% Now holdout is "Xhold, Yhold, holdoutStruct"
testInds = 1:size(Xhold,3);  % e.g. 1..(7*holdoutSize) ideally

%% 3) We'll store everything in a structure array allResults
allResults = struct();

%% 4) For i=1 to 7:
for i = 1:numAllDatasets
    fprintf('\n==== Using first %d set(s) for training. Testing on universal holdout. ====\n', i);

    % We'll store the metrics in #trainSizes x R arrays
    maeNormMatrix    = nan(numel(trainSizes), R);
    maeNmMatrix      = nan(numel(trainSizes), R);
    hertzMapeMatrix  = nan(numel(trainSizes), R);
    mod500MapeMatrix = nan(numel(trainSizes), R);

    % Build a local copy of the first i datasets for repeated draws
    % so that draws don't "use up" data in allData permanently across
    % different trainSizes. We do a "deep copy" of the leftover data.
    trainPool(i).datasets = copyDatasetStructs(allData(1:i)); 

    %% 4a) Loop over training sizes
    for sIdx = 1:numel(trainSizes)
        sVal = trainSizes(sIdx);
        fprintf('   - TrainSize = %d\n', sVal);

        for rDraw = 1:R
            % (i) Draw sVal curves from the FIRST i sets as evenly as possible
            dsCopy = copyDatasetStructs(trainPool(i).datasets); % make a fresh copy each draw
            if sum([dsCopy.N]) < sVal
                continue
            end
            [Xtrain, Ytrain] = drawEqualFromDatasets(dsCopy, sVal);

            % (ii) Optionally augment
            if enableAugment
                [XtrainAug, YtrainAug] = augmentData(Xtrain, Ytrain, 100);
                XtrainAll = XtrainAug;
                YtrainAll = YtrainAug;
            else
                XtrainAll = Xtrain;
                YtrainAll = Ytrain;
            end

            % (iii) Train
            trainedNet = trainModelCore(layers, XtrainAll, YtrainAll);
            if isempty(trainedNet)
                fprintf('     *Train failed => storing NaNs.\n');
                maeNormMatrix(sIdx,rDraw)    = NaN;
                maeNmMatrix(sIdx,rDraw)      = NaN;
                hertzMapeMatrix(sIdx,rDraw)  = NaN;
                mod500MapeMatrix(sIdx,rDraw) = NaN;
                continue;
            end

            % (iv) Evaluate on universal holdout
            metricsOut = testTrainedModelOnDataset_sub( ...
                trainedNet, Xhold, Yhold, holdoutStruct, testInds, indentationDepth_nm);

            maeNormMatrix(sIdx, rDraw)    = metricsOut.maeTestNorm;
            maeNmMatrix(sIdx, rDraw)      = metricsOut.maeTestNm;
            hertzMapeMatrix(sIdx, rDraw)  = metricsOut.hertzMAPE;
            mod500MapeMatrix(sIdx, rDraw) = metricsOut.mod500MAPE;
        end
    end

    %% 4b) Compute mean & std across R draws
    meanMaeNorm   = mean(maeNormMatrix, 2, 'omitnan');
    stdMaeNorm    = std(maeNormMatrix, 0, 2, 'omitnan');
    meanMaeNm     = mean(maeNmMatrix, 2, 'omitnan');
    stdMaeNm      = std(maeNmMatrix, 0, 2, 'omitnan');
    meanHertzMape = mean(hertzMapeMatrix, 2, 'omitnan');
    stdHertzMape  = std(hertzMapeMatrix, 0, 2, 'omitnan');
    mean500Mape   = mean(mod500MapeMatrix, 2, 'omitnan');
    std500Mape    = std(mod500MapeMatrix, 0, 2, 'omitnan');

    %% 4c) Store results
    allResults(i).iSets           = i;
    allResults(i).trainSizes      = trainSizes;
    allResults(i).maeNormMatrix   = maeNormMatrix;
    allResults(i).maeNmMatrix     = maeNmMatrix;
    allResults(i).hertzMapeMatrix = hertzMapeMatrix;
    allResults(i).mod500MapeMatrix= mod500MapeMatrix;

    allResults(i).meanMaeNorm   = meanMaeNorm;
    allResults(i).stdMaeNorm    = stdMaeNorm;
    allResults(i).meanMaeNm     = meanMaeNm;
    allResults(i).stdMaeNm      = stdMaeNm;
    allResults(i).meanHertzMape = meanHertzMape;
    allResults(i).stdHertzMape  = stdHertzMape;
    allResults(i).mean500Mape   = mean500Mape;
    allResults(i).std500Mape    = std500Mape;
end

%% 5) Plot results
figure('Name','MAE vs Train Size');
tiledlayout('vertical','TileSpacing','compact');
ax1 = nexttile(); hold on; grid on; title('MAE (Normalized)');
ax2 = nexttile(); hold on; grid on; title('MAE (nm)');

colors = lines(numAllDatasets);

for i = 1:numAllDatasets
    sVals = allResults(i).trainSizes;
    mMaeN = allResults(i).meanMaeNorm;
    eMaeN = allResults(i).stdMaeNorm;
    mMaeNm= allResults(i).meanMaeNm;
    eMaeNm= allResults(i).stdMaeNm;

    lbl = sprintf('%d set%s', i, pluralS(i));

    axes(ax1);
    errorbar(sVals, mMaeN, eMaeN, '-o','Color',colors(i,:),...
        'DisplayName',lbl);

    axes(ax2);
    errorbar(sVals, mMaeNm, eMaeNm, '-o','Color',colors(i,:),...
        'DisplayName',lbl);
end
legend(ax1,'Location','best');
legend(ax2,'Location','best');
xlabel(ax1,'Training set size'); ylabel(ax1,'MAE (norm)');
xlabel(ax2,'Training set size'); ylabel(ax2,'MAE (nm)');

figure('Name','Modulus MAPE vs Train Size');
tiledlayout('vertical','TileSpacing','compact');
ax3 = nexttile(); hold on; grid on; title('Hertz MAPE (%)');
ax4 = nexttile(); hold on; grid on; title('500 nm MAPE (%)');

for i = 1:numAllDatasets
    sVals    = allResults(i).trainSizes;
    mHertz   = allResults(i).meanHertzMape;
    eHertz   = allResults(i).stdHertzMape;
    m500     = allResults(i).mean500Mape;
    e500     = allResults(i).std500Mape;

    lbl = sprintf('%d set%s', i, pluralS(i));

    axes(ax3);
    errorbar(sVals, mHertz, eHertz, '-o','Color',colors(i,:),...
        'DisplayName',lbl);

    axes(ax4);
    errorbar(sVals, m500, e500, '-o','Color',colors(i,:),...
        'DisplayName',lbl);
end
legend(ax3,'Location','best');
legend(ax4,'Location','best');
xlabel(ax3,'Training set size'); ylabel(ax3,'MAPE (%)');
xlabel(ax4,'Training set size'); ylabel(ax4,'MAPE (%)');

%% Additional Single-Plot Figures
close all

% Define your start and end colors as [R G B], values between 0 and 1
% colorStart = [0.0 0.0 0.0];  % e.g. black
% colorEnd   = [1.0 0.0 0.0];  % e.g. red
% 
% colorStart = [ 0.7882    0.7882    0.0549];
% colorEnd = [  0.8196    0.0902    0.0902];
% 
% % Pre-allocate array
% colors = zeros(numAllDatasets, 3);
% colors = parula(numAllDatasets);
% 
% % Fill in each row of 'colors' by linear interpolation
% for i = 1:numAllDatasets
%     fraction = (i-1) / (numAllDatasets - 1);
%     colors(i,:) = colorStart + fraction * (colorEnd - colorStart);
% end

colors = copper(numAllDatasets);
colors = colors(end:-1:1,:);

% Now 'colors' is a Nx3 array where each row is a smooth step
% from colorStart to colorEnd.


% Single plot: Contact Point (MAE in nm) vs. Training Set Size
figure('Name','Learning Curve - Contact Point (nm)');
hold on; grid on;

for i = 1:numAllDatasets
    sVals  = allResults(i).trainSizes;
    mMaeNm = allResults(i).meanMaeNm;
    eMaeNm = allResults(i).stdMaeNm;

    lbl = sprintf('%d set%s', i, pluralS(i));

    errorbar(sVals, mMaeNm, eMaeNm, '-o', ...
        'Color', colors(i,:), ...
        'DisplayName', lbl);
end

xlabel('Training set size'); 
ylabel('Contact Point MAE (nm)');
title('Learning Curve - Contact Point MAE (Holdout = 100 per dataset)');
legend('Location','best');
xlim([0 2000])

% Single plot: Hertz MAPE (%) vs. Training Set Size
figure('Name','Learning Curve - Hertzian Modulus MAPE (Holdout = 100 per dataset)');
hold on; grid on;

for i = 1:numAllDatasets
    sVals  = allResults(i).trainSizes;
    mHertz = allResults(i).meanHertzMape;
    eHertz = allResults(i).stdHertzMape;

    lbl = sprintf('%d set%s', i, pluralS(i));

    errorbar(sVals, mHertz, eHertz, '-o', ...
        'Color', colors(i,:), ...
        'DisplayName', lbl);
end

xlabel('Training set size');
ylabel('Hertzian Modulus MAPE (%)');
title('Learning Curve - Hertzian Modulus MAPE (Holdout = 100 per dataset)');
legend('Location','best');

xlim([0 2000])


%% 6) Save
save('multiSourceResults_holdout.mat','allResults','numAllDatasets','cellTypeNames','holdoutStruct','Xhold','Yhold','testInds');
fprintf('\nAll done! Results (with universal holdout) saved in "multiSourceResults_holdout.mat".\n');


%% ------------------------------------------------------------------------
function dsCopy = copyDatasetStructs(dsArray)
% Utility to copy an array of dataset structs so that random draws
% won't permanently remove them from the original in each iteration.
dsCopy = dsArray; % shallow copy is usually fine for re-labelling
% But since we store arrays in X, Y, we just want a separate instance.
for k = 1:numel(dsArray)
    dsCopy(k).X = dsArray(k).X;
    dsCopy(k).Y = dsArray(k).Y;
    dsCopy(k).N = dsArray(k).N;
    dsCopy(k).minExtValues = dsArray(k).minExtValues;
    dsCopy(k).maxExtValues = dsArray(k).maxExtValues;
    dsCopy(k).rawExt       = dsArray(k).rawExt;
    dsCopy(k).rawDefl      = dsArray(k).rawDefl;
    dsCopy(k).goodOrBad    = dsArray(k).goodOrBad;
    dsCopy(k).b            = dsArray(k).b;
    dsCopy(k).th           = dsArray(k).th;
    dsCopy(k).R            = dsArray(k).R;
    dsCopy(k).v            = dsArray(k).v;
    dsCopy(k).spring_constant = dsArray(k).spring_constant;
end
end

%% ------------------------------------------------------------------------
function [Xtrain, Ytrain] = drawEqualFromDatasets(dataArray, sVal)
% Same logic as before: draw 'sVal' samples from these i datasets as evenly
% as possible, removing from .X/.Y so we can't pick them again next time.
i = numel(dataArray);
if i < 1
    Xtrain = [];
    Ytrain = [];
    return;
end

Xtrain = [];
Ytrain = [];
sRemaining = sVal;
dsRemaining = 1:i;

while sRemaining>0 && ~isempty(dsRemaining)
    nDatasets = numel(dsRemaining);
    baseTake  = floor(sRemaining/nDatasets);
    if baseTake<1
        baseTake=1;
    end

    toRemove = [];
    for k = dsRemaining
        if sRemaining<=0, break; end

        nAvail = dataArray(k).N;
        if nAvail<=0
            toRemove = [toRemove, k];
            continue;
        end

        takeAmount = baseTake;
        if takeAmount>nAvail
            takeAmount = nAvail;
        end

        % random pick
        permIdx = randperm(nAvail);
        useInds = permIdx(1:takeAmount);

        % Add them to Xtrain, Ytrain
        Xtrain = cat(3, Xtrain, dataArray(k).X(:,:, useInds));
        Ytrain = cat(1, Ytrain, dataArray(k).Y(useInds));

        % remove from dataset k
        dataArray(k).X(:,:, useInds) = [];
        dataArray(k).Y(useInds)      = [];
        dataArray(k).minExtValues(useInds) = [];
        dataArray(k).maxExtValues(useInds) = [];
        dataArray(k).rawExt(useInds)       = [];
        dataArray(k).rawDefl(useInds)      = [];
        dataArray(k).goodOrBad(useInds)    = [];
        dataArray(k).b(useInds)            = [];
        dataArray(k).th(useInds)           = [];
        dataArray(k).R(useInds)            = [];
        dataArray(k).v(useInds)            = [];
        dataArray(k).spring_constant(useInds) = [];

        dataArray(k).N = dataArray(k).N - takeAmount;
        sRemaining = sRemaining - takeAmount;

        if dataArray(k).N<=0
            toRemove = [toRemove, k];
        end

        if sRemaining<=0
            break;
        end
    end
    dsRemaining = setdiff(dsRemaining, toRemove);
end
end

%% ------------------------------------------------------------------------
function ds = initEmptyDataStruct()
% Creates an empty structure with the fields needed by testTrainedModelOnDataset_sub
ds = struct(...
    'minExtValues',[],...
    'maxExtValues',[],...
    'rawExt',[],...
    'rawDefl',[],...
    'goodOrBad',[],...
    'b',[],...
    'th',[],...
    'R',[],...
    'v',[],...
    'spring_constant',[]);
end

%% ------------------------------------------------------------------------
function s = pluralS(n)
% Utility to return 's' if n>1, else ''
if n>1, s='s'; else, s=''; end
end
