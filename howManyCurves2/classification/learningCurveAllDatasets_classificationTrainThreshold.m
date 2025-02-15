% This script performs a learning-curve analysis for classification data,
% but uses a fixed holdout portion (size=holdoutSize) for testing, never used in training.
%
% Steps:
%   1) For each dataset, randomly pick 'holdoutSize' curves => "universal" test set.
%   2) For each train-size sVal, do R repeated draws of sVal from the leftover portion => train => evaluate on holdout.
%   3) Collate mean ± std of metrics across R runs => learning curve.
%
% Usage:
%   learningCurveAllDatasets_classificationHoldout();
%     -> By default, holdoutSize=1000. Or pass a different holdoutSize.




clc; close all; clear;

%% 0) Setup

addpath('C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions')

% Example: classification data sets
dataSets = [
    "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat"
    % Add more file paths as needed, one per dataset
    ];
dataSetNames = ["All"]; % Provide matching names
numDatasets  = numel(dataSets);

% Train sizes to explore
trainSizes = [100:50:750, 1000:500:4500];

% Number of repeated random draws at each train size
R = 3;

% Classification CNN architecture
nFeatures      = 6;
sequenceLength = 2000;
layers = CNN_custom_pooling_after_lstm_2conv_relu_classification( ...
    nFeatures, sequenceLength, 7);

% We'll store final results in this struct
allResults = struct();

holdoutSize = 1000;
fprintf('Using a fixed holdout of %d samples from each dataset.\n', holdoutSize);


rng(1337);

%% 1) Loop over each dataset
for d = 1:numDatasets
    fprintf('\n=== LEARNING CURVE (Holdout) for dataset: %s ===\n', dataSetNames(d));

    dataFile = dataSets(d);
    S = load(dataFile, 'X','goodOrBad');
    if ~isfield(S,'X') || ~isfield(S,'goodOrBad')
        warning('File "%s" missing X or goodOrBad => skip.\n', dataFile);
        continue;
    end
    Xfull = S.X;             % [features x seqLen x N]
    Yfull = S.goodOrBad;     % [N x 1] or [1 x N], 0/1
    N     = size(Xfull,3);
    fprintf('  Found %d samples in %s.\n', N, dataFile);

    if N < 2
        warning('Not enough data => skip dataset.\n');
        continue;
    end

    % (a) Create a universal holdout of 'holdoutSize' from this dataset
    if holdoutSize >= N
        holdoutSize = N-1;
        fprintf('WARNING: holdoutSize >= N => adjusting holdout to %d\n', holdoutSize);
    end
    permAll = randperm(N);
    holdInds = permAll(1:holdoutSize);
    leftoverInds = permAll(holdoutSize+1:end);

    Xhold = Xfull(:,:, holdInds);
    Yhold = Yfull(holdInds);
    leftoverN = length(leftoverInds);

    fprintf('  Creating holdout of %d. Remainder for training: %d\n', holdoutSize, leftoverN);

    % Filter trainSizes => must not exceed leftoverN
    validSizes = trainSizes(trainSizes <= leftoverN);
    if isempty(validSizes)
        warning('No valid train sizes for dataset "%s" => skipping.\n', dataSetNames(d));
        continue;
    end

    % We'll store classification metrics => [#validSizes x R]
    accMatrix  = nan(numel(validSizes), R);
    recMatrix  = nan(numel(validSizes), R);
    precMatrix = nan(numel(validSizes), R);
    f1Matrix   = nan(numel(validSizes), R);
    aucMatrix  = nan(numel(validSizes), R);

    %% 2) For each train size
    for sIdx = 1:numel(validSizes)
        sVal = validSizes(sIdx);
        fprintf('  Train size = %d\n', sVal);

        for rDraw = 1:R
            % (i) Random subset of leftoverInds
            rp = randperm(leftoverN);
            pickTrain = rp(1:sVal);
            actualTrainInds = leftoverInds(pickTrain);

            % Build training set
            Xtrain = Xfull(:,:, actualTrainInds);
            Ytrain = Yfull(actualTrainInds);

            % Train
            trainedNet = trainModelCore_classification(layers, Xtrain, Ytrain);
            if isempty(trainedNet)
                accMatrix(sIdx, rDraw)  = NaN;
                recMatrix(sIdx, rDraw)  = NaN;
                precMatrix(sIdx, rDraw) = NaN;
                f1Matrix(sIdx, rDraw)   = NaN;
                aucMatrix(sIdx, rDraw)  = NaN;
                continue;
            end

            % Evaluate on the holdout
            metricsOut = testTrainedModel_classificationTrainOpt( ...
                trainedNet, Xtrain, Ytrain, Xhold, Yhold);

            accMatrix(sIdx, rDraw)  = metricsOut.accuracy;
            recMatrix(sIdx, rDraw)  = metricsOut.recall;
            precMatrix(sIdx, rDraw) = metricsOut.precision;
            f1Matrix(sIdx, rDraw)   = metricsOut.f1;
            aucMatrix(sIdx, rDraw)  = metricsOut.auc;
        end
    end

    %% 3) Compute mean & std across R runs
    meanAcc = mean(accMatrix,2,'omitnan');  stdAcc = std(accMatrix,0,2,'omitnan');
    meanRec = mean(recMatrix,2,'omitnan');  stdRec = std(recMatrix,0,2,'omitnan');
    meanPrec= mean(precMatrix,2,'omitnan'); stdPrec= std(precMatrix,0,2,'omitnan');
    meanF1  = mean(f1Matrix,2,'omitnan');   stdF1  = std(f1Matrix,0,2,'omitnan');
    meanAuc = mean(aucMatrix,2,'omitnan');  stdAuc = std(aucMatrix,0,2,'omitnan');

    %% 4) Store in allResults
    allResults(d).datasetName = dataSetNames(d);
    allResults(d).N           = N;
    allResults(d).trainSizes  = validSizes;

    allResults(d).accMatrix   = accMatrix;
    allResults(d).recMatrix   = recMatrix;
    allResults(d).precMatrix  = precMatrix;
    allResults(d).f1Matrix    = f1Matrix;
    allResults(d).aucMatrix   = aucMatrix;

    allResults(d).meanAcc  = meanAcc;  allResults(d).stdAcc  = stdAcc;
    allResults(d).meanRec  = meanRec;  allResults(d).stdRec  = stdRec;
    allResults(d).meanPrec = meanPrec; allResults(d).stdPrec = stdPrec;
    allResults(d).meanF1   = meanF1;   allResults(d).stdF1   = stdF1;
    allResults(d).meanAuc  = meanAuc;  allResults(d).stdAuc  = stdAuc;
end

%% 5) Plot example: Accuracy vs train size
figure('Name','Learning Curve - Accuracy (Fixed Holdout)');
hold on; grid on;
colors = lines(numDatasets);
for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end
    sVals = allResults(d).trainSizes;
    mAcc  = allResults(d).meanAcc;
    sAcc  = allResults(d).stdAcc;
    plotName = char(allResults(d).datasetName);

    errorbar(sVals, mAcc, sAcc, '-o', 'Color', colors(d,:), ...
        'DisplayName', plotName);
end
xlabel('Training Set Size');
ylabel('Accuracy');
legend('Location','best');
title(sprintf('Learning Curve - Accuracy (Holdout=%d)',holdoutSize));

%% 6) (Optional) Plot F1
figure('Name','Learning Curve - F1 (Fixed Holdout)');
hold on; grid on;
for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end
    sVals = allResults(d).trainSizes;
    mF1   = allResults(d).meanF1;
    sF1   = allResults(d).stdF1;
    plotName = char(allResults(d).datasetName);

    errorbar(sVals, mF1, sF1, '-o', 'Color', colors(d,:), ...
        'DisplayName', plotName);
end
xlabel('Training Set Size');
ylabel('F1 Score');
legend('Location','best');
title(sprintf('Learning Curve - F1 (Holdout=%d)',holdoutSize));

%% 7) Plot AUC
figure('Name','Learning Curve - AUC (Fixed Holdout)');
hold on; grid on;
for d = 1:numDatasets
    if ~isfield(allResults(d),'trainSizes') || isempty(allResults(d).trainSizes)
        continue;
    end
    sVals = allResults(d).trainSizes;
    mAuc   = allResults(d).meanAuc;
    sAuc   = allResults(d).stdAuc;
    plotName = char(allResults(d).datasetName);

    errorbar(sVals, mAuc, sAuc, '-o', 'Color','k', ...
        'DisplayName', plotName);
end
xlabel('Training Set Size');
ylabel('AUC');
legend('Location','best');
title(sprintf('Learning Curve - AUC (Holdout = %d)', holdoutSize));

%% 8) Save final results
saveFile = sprintf('learningCurveClassificationHoldout_%d.mat', holdoutSize);
save(saveFile,'allResults','numDatasets','colors','holdoutSize');
fprintf('\nAll done! Results saved in %s.\n', saveFile);



%% ------------------------------------------------------------------------
%%  HELPER: trainModelCore_classification
%% ------------------------------------------------------------------------
function trainedNet = trainModelCore_classification(layers, Xtrain, Ytrain)
% TRAINMODELCORE_CLASSIFICATION
%   A minimal classification training wrapper
%   Xtrain: [features x seqLen x N]
%   Ytrain: [N x 1] (0/1 or "reject"/"accept")

trainedNet = [];
N = size(Xtrain,3);
if N<2
    return;
end

% Convert labels => categorical
catY = toCategoricalRejectAccept(Ytrain);

% Convert X => 'CBT'
Xp  = permute(Xtrain,[1 3 2]);
dlX = dlarray(Xp,'CBT');

net = dlnetwork(layers);

opts = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize',64*.5, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','none', ...
    'ValidationFrequency',50, ...
    'InitialLearnRate',5e-4);

try
    [trainedNet, info] = trainnet(dlX, catY', net, "crossentropy", opts);
catch ME
    warning('Error in trainModelCore_classification => returning empty\n%s',ME.message);
    trainedNet = [];
end
end


%% ------------------------------------------------------------------------
%% HELPER: testTrainedModel_classificationTrainOpt
%% ------------------------------------------------------------------------
function metricsOut = testTrainedModel_classificationTrainOpt(net, Xtrain, Ytrain, Xtest, Ytest)
% 1) Predict train-set probabilities, pick optimal threshold
%    (min distance to (0,1) on ROC).
% 2) Evaluate on test set with that threshold.
% 3) AUC is threshold-free.

% Convert train labels => cat
catTrain = toCategoricalRejectAccept(Ytrain);
probTrain = predictOnData(net, Xtrain);
[fprTrain,tprTrain,thrTrain,~] = perfcurve(catTrain, probTrain(:,1), "reject");
dist2corner = sqrt((fprTrain - 0).^2 + (tprTrain - 1).^2);
[~, bestIdx] = min(dist2corner);
optThresh    = thrTrain(bestIdx);

% Test
catTest = toCategoricalRejectAccept(Ytest);
probTest = predictOnData(net, Xtest);
[~,~,~,aucVal] = perfcurve(catTest, probTest(:,1), "reject");

predIsReject = (probTest(:,1) >= optThresh);
strPred = repmat("accept", size(probTest,1),1);
strPred(predIsReject) = "reject";
catPred = categorical(strPred, ["reject","accept"]);

C = confusionmat(catTest, catPred);
TP = C(1,1); FN = C(1,2);
FP = C(2,1); TN = C(2,2);

recall    = TP/(TP+FN+eps);
precision = TP/(TP+FP+eps);
f1        = 2*(precision*recall)/(precision+recall+eps);
accuracy  = (TP+TN)/(TP+TN+FP+FN+eps);

metricsOut = struct('accuracy',accuracy,'recall',recall,'precision',precision,...
    'f1',f1,'auc',aucVal,'optThresh',optThresh);
end


%% ------------------------------------------------------------------------
%% HELPER: predictOnData
%% ------------------------------------------------------------------------
function probOut = predictOnData(net, X)
% PREDICTONDATA
Xp = permute(X,[1 3 2]);
dlX= dlarray(Xp,'CBT');
out= predict(net, dlX);
out2= gather(extractdata(out))';
probOut = out2;
end


%% ------------------------------------------------------------------------
%% HELPER: toCategoricalRejectAccept
%% ------------------------------------------------------------------------
function catLabels = toCategoricalRejectAccept(Y)
% Convert numeric => "reject","accept", or pass-thru if already cat
if isnumeric(Y)
    strY = strings(size(Y));
    strY(Y==0) = "reject";
    strY(Y==1) = "accept";
    catLabels = categorical(strY, ["reject","accept"]);
elseif isstring(Y) || ischar(Y) || iscellstr(Y)
    strY = string(Y);
    strY(strY=="0") = "reject";
    strY(strY=="1") = "accept";
    catLabels = categorical(strY, ["reject","accept"]);
elseif iscategorical(Y)
    if ~all(ismember(["reject","accept"], categories(Y)))
        newY = string(Y);
        newY(newY=="0") = "reject";
        newY(newY=="1") = "accept";
        catLabels = categorical(newY, ["reject","accept"]);
    else
        catLabels = Y;
    end
else
    error('Labels must be numeric, string, or categorical. Got: %s', class(Y));
end
end
