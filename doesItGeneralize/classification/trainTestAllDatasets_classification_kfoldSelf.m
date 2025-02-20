%% SCRIPT: trainTestAllDatasets_classification_kfoldSelf.m
%
% Trains exactly one classification model per dataset for cross-dataset usage.
% Then:
%   - If trainIdx ~= testIdx => We test on the entire target dataset.
%   - If trainIdx == testIdx => We do a K-fold cross-validation (k=5)
%     on the entire same dataset, ignoring leftover splits.
%
% This version removes the old "80/20 leftover" logic for smaller datasets;
% if N < numSamplesToTrain, it trains on ALL curves. No leftover for self-test.
%

clear; clc; close all;

%% 1) Setup
k = 5;  % number of folds for self-test
dataSets = [
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_LM24.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_MCF7.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_MCF10a.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_HEPG4.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_podocytes.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_tubules.mat"
        "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_iPSC_VSMC.mat"
];

dataSetNames = ["LM24","MCF7","MCF10a","HEPG4","Podocytes","Tubules","iPSC VSMC"];
numDatasets  = numel(dataSets);

addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions");

% Number of samples to train from each dataset (for cross-dataset usage).
numSamplesToTrain = 10000;  % If dataset N < 10000 => train on all curves

% CNN architecture
nFeatures      = 6;
sequenceLength = 2000;
layers = CNN_custom_pooling_after_bilstm_2conv_relu_classification(nFeatures, sequenceLength, 7);

% Where to store results
resultsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\doesItGeneralize\v3_classification\bilstm_classificationResults";
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% If you want a different threshold, set it here (numeric or 'optimal')
classificationThreshold = 'optimal';

%% 2) Result matrices
accuracyMatrix = nan(numDatasets,numDatasets);
aucMatrix      = nan(numDatasets,numDatasets);
recallMatrix   = nan(numDatasets,numDatasets);

% We'll store the model (for cross-dataset) in a struct
trainedInfo = cell(numDatasets,1);

%% 3) Train each dataset once (for cross-dataset usage)
for trainIdx = 1:numDatasets
    fprintf('\n=== TRAINING on %s (%s) ===\n', ...
        dataSetNames(trainIdx), dataSets(trainIdx));

    [model] = trainClassificationOneDataset_singleModel_noLeftover( ...
        layers, dataSets(trainIdx), numSamplesToTrain);

    % Save the model if you want
    modelSaveName = fullfile(resultsFolder, ...
        sprintf('trainedModel_%s.mat', dataSetNames(trainIdx)));
    save(modelSaveName, 'model');

    % Store in memory
    infoStruct.model = model;
    trainedInfo{trainIdx} = infoStruct;
end

%% 4) Test each trained model on each dataset
for trainIdx = 1:numDatasets
    infoStruct = trainedInfo{trainIdx};
    if isempty(infoStruct) || isempty(infoStruct.model)
        warning('No trained model for dataset %s => skipping tests', dataSetNames(trainIdx));
        continue;
    end

    thisNet = infoStruct.model;

    for testIdx = 1:numDatasets
        fprintf('\n--- TESTING model from %s on dataset %s ---\n', ...
            dataSetNames(trainIdx), dataSetNames(testIdx));

        testSaveName = fullfile(resultsFolder, ...
            sprintf('train_%s_test_%s.mat', ...
            dataSetNames(trainIdx), dataSetNames(testIdx)));

        if trainIdx == testIdx
            % SELF-TEST => K-fold crossvalidation (k=5) on the entire dataset
            fprintf('Self-test => using %d-fold crossvalidation on the same data.\n', k);
            metricsCV = testKfoldSameDataset_classification(layers, dataSets(testIdx), classificationThreshold, k);

            accuracyMatrix(trainIdx,testIdx) = metricsCV.accuracy;
            aucMatrix(trainIdx,testIdx)      = metricsCV.auc;
            recallMatrix(trainIdx,testIdx)   = metricsCV.recall;

            fprintf('Self-test (k-fold) => Acc=%.2f%%, AUC=%.3f, R=%.3f\n', ...
                100*metricsCV.accuracy, metricsCV.auc, metricsCV.recall);
        else
            % CROSS-DATASET => use the single model on the entire test dataset
            dataTest = load(dataSets(testIdx), 'X','goodOrBad');
            if ~isfield(dataTest,'X') || ~isfield(dataTest,'goodOrBad')
                warning('Data file missing X/goodOrBad => skipping');
                accuracyMatrix(trainIdx,testIdx) = NaN;
                aucMatrix(trainIdx,testIdx)      = NaN;
                recallMatrix(trainIdx,testIdx)   = NaN;
                continue;
            end

            XtestAll = dataTest.X;
            YtestAll = dataTest.goodOrBad;
            if isempty(XtestAll)
                accuracyMatrix(trainIdx,testIdx) = NaN;
                aucMatrix(trainIdx,testIdx)      = NaN;
                recallMatrix(trainIdx,testIdx)   = NaN;
                continue;
            end

            % Evaluate
            metricsOut = testTrainedModelOnDataset_classification( ...
                thisNet, XtestAll, YtestAll, ...
                classificationThreshold, true, testSaveName);

            accuracyMatrix(trainIdx,testIdx) = metricsOut.accuracy;
            aucMatrix(trainIdx,testIdx)      = metricsOut.auc;
            recallMatrix(trainIdx,testIdx)   = metricsOut.recall;

            fprintf('Accuracy=%.2f%%, AUC=%.3f, Recall=%.3f\n', ...
                100*metricsOut.accuracy, metricsOut.auc, metricsOut.recall);
        end
    end
end

%% 5) Show Heatmaps
%close all

fontSize = 18;
fontSizeLegend = 13;
fontFamily = 'Arial';
cellLabelColorStr = 'auto';



% Reorder 
newIdx = [1 3 4 2 6 5 7];
dataSetNames2 = dataSetNames(newIdx);
aucMatrix2 = aucMatrix(newIdx, newIdx);



%figure;

% heatmap(dataSetNames, dataSetNames, accuracyMatrix, ...
%     'Title','Classification Accuracy', ...
%     'Colormap', magma, 'CellLabelColor',cellLabelColorStr);
% xlabel('Testing Dataset'); ylabel('Training Dataset');

figure('Name','AUC', ...
       'Units','inches', 'Position',[1 1 5 5]*1.3);
set(gca,'FontName',fontFamily,'FontSize',fontSize)
h = heatmap(dataSetNames2, dataSetNames2, round(aucMatrix2,2), ...
    'Colormap', magma, 'CellLabelColor',cellLabelColorStr,'FontSize', fontSize, ...
    'FontName',fontFamily);
h.Position = [0.26 0.36 0.60 0.60];
set(struct(h).NodeChildren(3), 'YTickLabelRotation', 45);
set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45); 



xlabel('Testing Dataset'); ylabel('Training Dataset');


%figure;
% heatmap(dataSetNames, dataSetNames, recallMatrix, ...
%     'Title','Classification Recall', ...
%     'Colormap', magma, 'CellLabelColor',cellLabelColorStr);
% xlabel('Testing Dataset'); ylabel('Training Dataset');
%% Save
save('bilstm_classification_results_does_it_generalize.mat', 'cellLabelColorStr', ...
    'accuracyMatrix','aucMatrix','k','dataSetNames')
disp('Classification train/test script (k-fold for self) completed successfully!');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER: trainClassificationOneDataset_singleModel_noLeftover
%%  => trains on min(N, numSamplesToTrain), no leftover
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainedNet = trainClassificationOneDataset_singleModel_noLeftover(layers, dataFile, numSamplesToTrain)
% TRAINCLASSIFICATIONONEDATASET_SINGLEMODEL_NOLEFTOVER
% 1) Loads dataFile => X, goodOrBad
% 2) If #samples >= numSamplesToTrain => pick exactly that many for train
%    If #samples <  numSamplesToTrain => use ALL (N) for train
% 3) Train a single CNN model
% 4) Return the trained net. No leftover is created.

    trainedNet = [];
    fprintf('Loading data from "%s"...\n', dataFile);

    S = load(dataFile, 'X','goodOrBad');
    if ~isfield(S,'X') || ~isfield(S,'goodOrBad')
        warning('Data file missing X/goodOrBad => returning empty.');
        return;
    end

    Xfull = S.X;
    Yfull = S.goodOrBad;
    N     = size(Xfull,3);
    fprintf('Dataset has %d curves.\n', N);

    if N < 2
        warning('Not enough data => returning empty model.');
        return;
    end

    rng(1337,'twister');  % reproducible

    % If dataset smaller than numSamplesToTrain => we just use all
    sVal = min(N, numSamplesToTrain);

    % random subset of sVal from the dataset
    perm = randperm(N);
    trainInd = perm(1:sVal);
    fprintf('Using %d for train (no leftover)\n', sVal);

    Xtrain = Xfull(:,:,trainInd);
    Ytrain = Yfull(trainInd);

    % Convert 0=>reject,1=>accept => categorical
    strTrain = strings(sVal,1);
    strTrain(Ytrain==0)="reject";
    strTrain(Ytrain==1)="accept";
    YcatTrain = categorical(strTrain, {'reject','accept'});

    net = dlnetwork(layers);

    % permute => 'CBT'
    XtrainPerm = permute(Xtrain,[1,3,2]);
    dlXtrain   = dlarray(XtrainPerm,'CBT');

    % shuffle
    rp = randperm(size(dlXtrain,2));
    dlXtrain = dlXtrain(:,rp,:);
    YcatTrain= YcatTrain(rp);

    opts = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64*.5,...
        'Shuffle','every-epoch',...
        'Verbose',true,...
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',1e-4);

    fprintf('Training single CNN model (no leftover logic)...\n');
    try
        [trainedNet,info] = trainnet(dlXtrain, YcatTrain, net, "crossentropy", opts);
        disp('Training completed.');
    catch ME
        warning('Error training: %s => returning empty model.', ME.message);
        trainedNet=[];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER: testKfoldSameDataset_classification
%% => standard k-fold CV for classification, averaging final metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function metricsMean = testKfoldSameDataset_classification(layers, dataFile, threshold, k)
% TESTKFOLDSAMEDATASET_CLASSIFICATION
%   Loads the entire classification dataset => k-fold split => for each fold,
%   train a new net on the (k-1) folds => test on the 1 fold => measure
%   accuracy, AUC, recall, etc. => average across folds.
%
% Inputs:
%   layers   -> CNN architecture
%   dataFile -> .mat with X, goodOrBad
%   threshold-> numeric or 'optimal'
%   k        -> #folds (e.g. 5)
%
% Output:
%   metricsMean => struct with .accuracy, .auc, .recall, .precision, .f1

    D = load(dataFile, 'X','goodOrBad');
    if ~isfield(D,'X') || ~isfield(D,'goodOrBad')
        warning('Missing X/goodOrBad => returning NaNs.');
        metricsMean = struct('accuracy',NaN,'auc',NaN,'recall',NaN,'precision',NaN,'f1',NaN);
        return;
    end

    Xall = D.X;
    Yall = D.goodOrBad;
    N    = size(Xall,3);

    if N < k
        warning('Not enough samples to do %d-fold => returning NaNs.', k);
        metricsMean = struct('accuracy',NaN,'auc',NaN,'recall',NaN,'precision',NaN,'f1',NaN);
        return;
    end

    indices = crossvalind('Kfold', N, k);

    accAll   = nan(k,1);
    aucAll   = nan(k,1);
    recAll   = nan(k,1);
    precAll  = nan(k,1);
    f1All    = nan(k,1);

    for foldID = 1:k
        fprintf('  K-fold: fold %d of %d\n', foldID, k);

        testMask  = (indices == foldID);
        trainMask = ~testMask;

        XtrainFold = Xall(:,:, trainMask);
        YtrainFold = Yall(trainMask);

        XtestFold  = Xall(:,:, testMask);
        YtestFold  = Yall(testMask);

        % Train a fold model:
        netFold = trainFoldCNN(layers, XtrainFold, YtrainFold);

        if isempty(netFold)
            warning('Fold %d: training returned empty net => skip metrics.', foldID);
            continue;
        end

        % Evaluate on the fold's test partition
        metricsFold = testTrainedModelOnDataset_classification( ...
            netFold, XtestFold, YtestFold, threshold, false, '');

        accAll(foldID)   = metricsFold.accuracy;
        aucAll(foldID)   = metricsFold.auc;
        recAll(foldID)   = metricsFold.recall;
        precAll(foldID)  = metricsFold.precision;
        f1All(foldID)    = metricsFold.f1;
    end

    % average across folds
    metricsMean = struct();
    metricsMean.accuracy  = mean(accAll,'omitnan');
    metricsMean.auc       = mean(aucAll,'omitnan');
    metricsMean.recall    = mean(recAll,'omitnan');
    metricsMean.precision = mean(precAll,'omitnan');
    metricsMean.f1        = mean(f1All,'omitnan');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER: trainFoldCNN => per-fold classification training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function netFold = trainFoldCNN(layers, Xtrain, Ytrain)
% TRAINFOLDCNN
%   Minimal training for each k-fold, no leftover or partial logic.

    if isempty(Xtrain) || isempty(Ytrain)
        netFold = [];
        return;
    end

    nTrain = numel(Ytrain);
    strTrain = strings(nTrain,1);
    strTrain(Ytrain==0) = "reject";
    strTrain(Ytrain==1) = "accept";
    YcatTrain = categorical(strTrain, {'reject','accept'});

    if nTrain < 2
        warning('Fold has <2 => returning empty net.');
        netFold = [];
        return;
    end

    netDL = dlnetwork(layers);

    Xperm   = permute(Xtrain,[1,3,2]); % => [F x N x S]
    dlX     = dlarray(Xperm,'CBT');
    rp      = randperm(size(dlX,2));
    dlX     = dlX(:,rp,:);
    YcatTrain = YcatTrain(rp);

    opts = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64*.5,...
        'Shuffle','every-epoch',...
        'Verbose', true,...  % <-- set to true to see iteration logs in each fold
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',1e-4);

    try
        [netFold,info] = trainnet(dlX, YcatTrain, netDL, "crossentropy", opts);
    catch ME
        warning('Error training fold => returning empty net: %s', ME.message);
        netFold = [];
    end
end
