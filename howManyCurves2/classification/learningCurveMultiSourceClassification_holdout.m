
% LEARNINGCURVEMULTISOURCECLASSIFICATION_HOLDOUT
%
% A multi-dataset learning-curve analysis for binary classification, using:
%   1) A single universal holdout of "holdoutSize" curves from EACH dataset (excluded from training).
%   2) Combine the first i sets (i=1..7) for training draws of sVal, then test on the same universal holdout.
%   3) Classification metrics: accuracy, recall, precision, F1, AUC.
%   4) Threshold for classification is derived from training data's ROC (min dist to (0,1)).
%
% By default, we treat "goodOrBad" as 0/1 labels. If it's a different variable name, adapt as needed.
%
% Data File Requirements:
%   - Each .mat must contain X ( [features x seqLen x N] ) and goodOrBad ( [N x 1] or [1 x N] ), etc.
%
% The order of the data sets is:
%   1) tubules
%   2) podocytes
%   3) HEPG4
%   4) iPSC
%   5) LM24
%   6) MCF7
%   7) MCF10a
%
% Usage: Just run the script. Modify architecture, train sizes, #repeats as desired.

clear; clc; close all;

%% 0) Data paths in the EXACT order you requested (classification .mat files)

addpath('C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions')

dataPaths = [
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_tubules.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_podocytes.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_HEPG4.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_iPSC_VSMC.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_LM24.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_MCF7.mat"
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_MCF10a.mat"
    ];

cellTypeNames = [ ...
    "tubules","podocytes","HEPG4","iPSC","LM24","MCF7","MCF10a" ...
    ];
numAllDatasets = numel(dataPaths);

%% (a) You can change holdoutSize if you want
holdoutSize = 100;   % # curves per dataset for the universal holdout

% Example training sizes
trainSizes = [100:100:1000, 1500, 2000, 2500, 3000];
R = 5;  % # repeated random draws

enableAugment = false;   % set true if you want data augmentation

%% (b) Architecture: pick your classification network
% e.g. a CNN with 2 output classes: "reject" (0) vs "accept" (1)
nFeatures = 6;
seqLength = 2000;
layers = CNN_custom_pooling_after_bilstm_2conv_relu_classification(nFeatures, seqLength, 7);

%% 1) Load all data into a struct array
allData = struct();
for idx = 1:numAllDatasets
    S = load(dataPaths(idx), 'X','goodOrBad');
    X = S.X;   % [features x seqLen x N]
    Y = S.goodOrBad;  % [N x 1] or [1 x N]
    if isrow(Y), Y = Y(:); end  % ensure Nx1
    allData(idx).X = X;
    allData(idx).Y = Y;
    allData(idx).N = size(X,3);
    allData(idx).name = cellTypeNames(idx);
    fprintf('Loaded dataset "%s": %d curves.\n', cellTypeNames(idx), allData(idx).N);
end

%% 2) Create the universal holdout of 'holdoutSize' from each dataset
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
    % Add to universal holdout
    Xhold = cat(3, Xhold, Xh);
    Yhold = [Yhold; Yh];  %#ok<AGROW>
    % Remove holdout samples from training data
    allData(idx).X(:,:, holdInds) = [];
    allData(idx).Y(holdInds)      = [];
    allData(idx).N = dsN - hSizeReal;
    fprintf('  Dataset %s: held out %d, leftover for train=%d.\n', ...
        allData(idx).name, hSizeReal, allData(idx).N);
end

% Universal test set indices (if needed later)
testInds = 1:size(Xhold,3);

%% 3) We'll store everything in a structure array allResults
allResults = struct();

%% 4) For i=1 to numAllDatasets:
for i = 1:numAllDatasets
    fprintf('\n==== Using first %d set(s) for training. Testing on universal holdout. ====\n', i);
    % We'll store the classification metrics in [#trainSizes x R] arrays
    accMatrix  = nan(numel(trainSizes), R);
    recMatrix  = nan(numel(trainSizes), R);
    precMatrix = nan(numel(trainSizes), R);
    f1Matrix   = nan(numel(trainSizes), R);
    aucMatrix  = nan(numel(trainSizes), R);

    % Make a local copy of the first i datasets, so draws don't remove from allData permanently
    trainPool(i).datasets = copyDatasetStructs(allData(1:i)); %#ok<AGROW>

    %% 4a) Loop over training sizes
    for sIdx = 1:numel(trainSizes)
        sVal = trainSizes(sIdx);
        % Compute total leftover in the local copy:
        totalLeftover = 0;
        for k = 1:numel(trainPool(i).datasets)
            totalLeftover = totalLeftover + trainPool(i).datasets(k).N;
        end
        if totalLeftover < sVal
            fprintf('   Not enough leftover (%d) for sVal=%d => skipping this training size.\n', totalLeftover, sVal);
            continue;
        end

        fprintf('   - TrainSize = %d\n', sVal);
        for rDraw = 1:R
            % (i) Draw sVal curves from the first i datasets evenly
            dsCopy = copyDatasetStructs(trainPool(i).datasets);
            [Xtrain, Ytrain] = drawEqualFromDatasets(dsCopy, sVal);
            % (ii) Optionally augment (if enabled)
            if enableAugment
                [XtrainAll, YtrainAll] = augmentDataClassification(Xtrain, Ytrain);
            else
                XtrainAll = Xtrain;
                YtrainAll = Ytrain;
            end
            % (iii) Train classification model
            trainedNet = trainModelCore_classification(layers, XtrainAll, YtrainAll);
            if isempty(trainedNet)
                accMatrix(sIdx,rDraw)  = NaN;
                recMatrix(sIdx,rDraw)  = NaN;
                precMatrix(sIdx,rDraw) = NaN;
                f1Matrix(sIdx,rDraw)   = NaN;
                aucMatrix(sIdx,rDraw)  = NaN;
                continue;
            end
            % (iv) Evaluate on universal holdout using train-based threshold
            metricsOut = testTrainedModel_classificationTrainOpt( ...
                trainedNet, XtrainAll, YtrainAll, Xhold, Yhold);
            accMatrix(sIdx,rDraw)  = metricsOut.accuracy;
            recMatrix(sIdx,rDraw)  = metricsOut.recall;
            precMatrix(sIdx,rDraw) = metricsOut.precision;
            f1Matrix(sIdx,rDraw)   = metricsOut.f1;
            aucMatrix(sIdx,rDraw)  = metricsOut.auc;
        end
    end

    %% 4b) Compute mean & std across R runs
    meanAcc  = mean(accMatrix,2,'omitnan');  stdAcc  = std(accMatrix,0,2,'omitnan');
    meanRec  = mean(recMatrix,2,'omitnan');  stdRec  = std(recMatrix,0,2,'omitnan');
    meanPrec = mean(precMatrix,2,'omitnan'); stdPrec = std(precMatrix,0,2,'omitnan');
    meanF1   = mean(f1Matrix,2,'omitnan');   stdF1   = std(f1Matrix,0,2,'omitnan');
    meanAuc  = mean(aucMatrix,2,'omitnan');  stdAuc  = std(aucMatrix,0,2,'omitnan');

    %% 4c) Store results
    allResults(i).iSets      = i;
    allResults(i).trainSizes = trainSizes;
    allResults(i).accMatrix  = accMatrix;
    allResults(i).recMatrix  = recMatrix;
    allResults(i).precMatrix = precMatrix;
    allResults(i).f1Matrix   = f1Matrix;
    allResults(i).aucMatrix  = aucMatrix;
    allResults(i).meanAcc    = meanAcc;  allResults(i).stdAcc  = stdAcc;
    allResults(i).meanRec    = meanRec;  allResults(i).stdRec  = stdRec;
    allResults(i).meanPrec   = meanPrec; allResults(i).stdPrec = stdPrec;
    allResults(i).meanF1     = meanF1;   allResults(i).stdF1   = stdF1;
    allResults(i).meanAuc    = meanAuc;  allResults(i).stdAuc  = stdAuc;
end

%% 5) Plot example: Accuracy & F1
figure('Name','Classification Learning Curves: Accuracy & F1');
tiledlayout('vertical','TileSpacing','compact');
ax1 = nexttile(); hold on; grid on; title('Accuracy');
ax2 = nexttile(); hold on; grid on; title('F1 Score');
colors = copper(numAllDatasets);
for i = 1:numAllDatasets
    sVals = allResults(i).trainSizes;
    mAcc  = allResults(i).meanAcc;
    eAcc  = allResults(i).stdAcc;
    mF1   = allResults(i).meanF1;
    eF1   = allResults(i).stdF1;
    lbl = sprintf('%d set%s', i, pluralS(i));
    axes(ax1);
    errorbar(sVals, mAcc, eAcc, '-o','Color',colors(i,:), 'DisplayName',lbl);
    axes(ax2);
    errorbar(sVals, mF1, eF1, '-o','Color',colors(i,:), 'DisplayName',lbl);
end
legend(ax1,'Location','best'); legend(ax2,'Location','best');
xlabel(ax1,'Training set size'); ylabel(ax1,'Accuracy');
xlabel(ax2,'Training set size'); ylabel(ax2,'F1');

%% 6) Plot AUC

fontSize = 18;
fontSizeLegend = 13;
fontFamily = 'Arial';
cellLabelColorStr = 'auto';
gridAlpha = 0.3;
lineWidth = 1.5;

% Create figure with specified size (6" x 4.5") and use inches as units
figure('Name','Classification Learning Curves: AUC', ...
    'Units','inches', 'Position',[1 1 5 5]*1.3);
hold on; grid on;

% Set axes font properties: Arial and font size 16
set(gca, 'FontName', fontFamily, 'FontSize', fontSize);

% Choose colors and reverse order
colors = copper(numAllDatasets);
colors = colors(end:-1:1,:);

for i = 1:numAllDatasets
    sVals = allResults(i).trainSizes;
    mAuc  = allResults(i).meanAuc;
    eAuc  = allResults(i).stdAuc;
    lbl = sprintf('%d set%s', i, pluralS(i));
    % Plot error bars with line width 1.5 and marker style '-o'
    errorbar(sVals, mAuc, eAuc, '-o', 'Color', colors(i,:), ...
             'DisplayName', lbl, 'LineWidth', lineWidth);
end

set(gca, 'Position', [0.26 0.36 0.60 0.60]);
set(gca,'GridAlpha',gridAlpha)
set(gca,'FontSize',fontSize,'FontName',fontFamily)
set(gca,'LineWidth',lineWidth)


% Set labels with Arial and font size 16
xlabel('Training set size')
ylabel('AUC')

% Create legend with font size 13; no title is added
legend('Location','best','FontSize',13);

set(gca,'GridAlpha',gridAlpha)
% Set axis limits
xlim([0 2000]);
ylim([0.54 0.91]);



%% 7) Save final results
save('bilstm_multiSourceClassificationResults_holdout.mat','allResults','numAllDatasets','cellTypeNames','Xhold','Yhold','testInds');
fprintf('\nAll done! Results (with universal classification holdout) saved in "multiSourceClassificationResults_holdout.mat".\n');


%% ------------------------------------------------------------------------
function dsCopy = copyDatasetStructs(dsArray)
dsCopy = dsArray;
for k = 1:numel(dsArray)
    dsCopy(k).X = dsArray(k).X;
    dsCopy(k).Y = dsArray(k).Y;
    dsCopy(k).N = dsArray(k).N;
end
end

%% ------------------------------------------------------------------------
function [Xtrain, Ytrain] = drawEqualFromDatasets(dataArray, sVal)
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
while sRemaining > 0 && ~isempty(dsRemaining)
    nDatasets = numel(dsRemaining);
    baseTake  = floor(sRemaining/nDatasets);
    if baseTake < 1, baseTake = 1; end
    toRemove = [];
    for k = dsRemaining
        if sRemaining <= 0, break; end
        nAvail = dataArray(k).N;
        if nAvail <= 0
            toRemove = [toRemove, k];
            continue;
        end
        takeAmount = baseTake;
        if takeAmount > nAvail
            takeAmount = nAvail;
        end
        permIdx = randperm(nAvail);
        useInds = permIdx(1:takeAmount);
        Xtrain = cat(3, Xtrain, dataArray(k).X(:,:, useInds));
        Ytrain = [Ytrain; dataArray(k).Y(useInds)];
        dataArray(k).X(:,:, useInds) = [];
        dataArray(k).Y(useInds) = [];
        dataArray(k).N = dataArray(k).N - takeAmount;
        sRemaining = sRemaining - takeAmount;
        if dataArray(k).N <= 0
            toRemove = [toRemove, k];
        end
        if sRemaining <= 0
            break;
        end
    end
    dsRemaining = setdiff(dsRemaining, toRemove);
end
end

%% ------------------------------------------------------------------------
function s = pluralS(n)
if n > 1, s = 's'; else, s = ''; end
end

%% ------------------------------------------------------------------------
function [Xaug, Yaug] = augmentDataClassification(Xin, Yin)
Xaug = Xin;
Yaug = Yin;
end

%% ------------------------------------------------------------------------
function metricsOut = testTrainedModel_classificationTrainOpt(net, Xtrain, Ytrain, Xtest, Ytest)
catTrain = toCategoricalRejectAccept(Ytrain);
catTest  = toCategoricalRejectAccept(Ytest);
probTrain = predictOnData(net, Xtrain);
[fprTrain, tprTrain, thrTrain, ~] = perfcurve(catTrain, probTrain(:,1), "reject");
dist2corner = sqrt((fprTrain - 0).^2 + (tprTrain - 1).^2);
[~, bestIdx] = min(dist2corner);
optThresh = thrTrain(bestIdx);
probTest = predictOnData(net, Xtest);
[~,~,~,aucTest] = perfcurve(catTest, probTest(:,1), "reject");
predIsReject = (probTest(:,1) >= optThresh);
predLabels = repmat("accept", size(probTest,1), 1);
predLabels(predIsReject) = "reject";
catPred = categorical(predLabels, ["reject", "accept"]);
C = confusionmat(catTest, catPred);
TP = C(1,1); FN = C(1,2);
FP = C(2,1); TN = C(2,2);
recall = TP/(TP+FN+eps);
precision = TP/(TP+FP+eps);
f1 = 2*(precision*recall)/(precision+recall+eps);
accuracy = (TP+TN)/(TP+TN+FP+FN+eps);
metricsOut = struct('accuracy', accuracy, 'recall', recall, 'precision', precision, ...
    'f1', f1, 'auc', aucTest, 'optThresh', optThresh);
end

%% ------------------------------------------------------------------------
function probOut = predictOnData(net, X)
Xp = permute(X, [1 3 2]);
dlX = dlarray(Xp, 'CBT');
out = predict(net, dlX);
out2 = gather(extractdata(out))';
probOut = out2;
end

%% ------------------------------------------------------------------------
function catLabels = toCategoricalRejectAccept(Y)
if isnumeric(Y)
    strY = strings(size(Y));
    strY(Y==0) = "reject";
    strY(Y==1) = "accept";
    catLabels = categorical(strY, ["reject", "accept"]);
elseif isstring(Y) || ischar(Y) || iscellstr(Y)
    strY = string(Y);
    strY(strY=="0") = "reject";
    strY(strY=="1") = "accept";
    catLabels = categorical(strY, ["reject", "accept"]);
elseif iscategorical(Y)
    if ~all(ismember(["reject", "accept"], categories(Y)))
        newY = string(Y);
        newY(newY=="0") = "reject";
        newY(newY=="1") = "accept";
        catLabels = categorical(newY, ["reject", "accept"]);
    else
        catLabels = Y;
    end
else
    error('Labels must be numeric, string, or categorical. Got: %s', class(Y));
end
end
