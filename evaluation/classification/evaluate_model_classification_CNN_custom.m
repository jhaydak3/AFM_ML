function evaluate_model_classification_CNN_custom(layers, preprocessedDataFile, saveName, useAllFeatures)
%EVALUATE_MODEL_CLASSIFICATION_CNN_CUSTOM
% Performs k-fold cross-validation for binary classification using a custom CNN model.
% Calculates classification metrics (accuracy, recall, precision, F1-score, TP, FP, FN, TN, and AUC).
% Plots ROC curves for each fold. Also creates a CSV of misclassified test
% samples for each fold if 'fileRow','fileCol','fileIndices' exist.
%
% In addition, for each fold:
%   (1) We compute the ROC on the TRAINING set, find the threshold that
%       minimizes distance to (0,1).
%   (2) We apply that train-based optimal threshold to the TEST predictions
%       (probabilities) and compute final metrics (optimalRecall, etc.).
%   (3) We store & report them, so we do NOT pick threshold from the test set.
%
% INPUTS:
%   layers  - Layer array for custom CNN
%   preprocessedDataFile - .mat file with X, goodOrBad, optional fileRow,fileCol,fileIndices
%   saveName - .mat filename to save results
%   useAllFeatures - if true => use all X features, else X(1,:,:)
%
% OUTPUT:
%   None (results are saved to disk)

clc;
close all;
fprintf('Starting k-fold cross-validation using Custom CNN Classification Model...\n');

%% ==================== Configuration ==================== %%
k = 5;
rng(1337,'twister'); % reproducible

% Add helper path if needed
helperFunctionsFolder = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v4_CNI_predict\helperFunctions";
addpath(helperFunctionsFolder);

%% ==================== Load Data ==================== %%
fprintf('Loading preprocessed data from "%s"...\n', preprocessedDataFile);
data = load(preprocessedDataFile);

% Basic checks
requiredFieldsPD = {'X','goodOrBad'};
for i = 1:numel(requiredFieldsPD)
    if ~isfield(data, requiredFieldsPD{i})
        error('Field "%s" missing in "%s".', requiredFieldsPD{i}, preprocessedDataFile);
    end
end

% Check for optional fields to record misclassifications
haveRow     = isfield(data,'fileRow');
haveCol     = isfield(data,'fileCol');
haveIndices = isfield(data,'fileIndices');

X = data.X;
goodOrBad = data.goodOrBad;  % numeric or string-based

numSamples = size(X,3);
fprintf('Loaded %d samples.\n', numSamples);

%% Convert numeric 0/1 => 'reject','accept', if needed
if isnumeric(goodOrBad)
    goodOrBad2 = strings(size(goodOrBad));
    goodOrBad2(goodOrBad==0) = "reject";
    goodOrBad2(goodOrBad==1) = "accept";
else
    % If string-based '0' => 'reject', etc.
    goodOrBad2 = strings(size(goodOrBad));
    goodOrBad2(goodOrBad=='0')="reject";
    goodOrBad2(goodOrBad=='1')="accept";
end
goodOrBad2 = categorical(goodOrBad2,{'reject','accept'});

%% ==================== Initialize Performance Arrays ==================== %%
accuracyScores     = zeros(k,1);
trainAccuracyScores= zeros(k,1);
recallScores       = zeros(k,1);
precisionScores    = zeros(k,1);
f1Scores           = zeros(k,1);
aucScores          = zeros(k,1);
aucScoresTrain     = zeros(k,1);

trainRecallScores = zeros(k,1);
trainPrecisionScores = zeros(k,1);
trainF1Scores     = zeros(k,1);

% For storing confusion sums
TPTrain = zeros(k,1); FPTrain=zeros(k,1);
FNTrain = zeros(k,1); TNTrain=zeros(k,1);
TPTest  = zeros(k,1); FPTest=zeros(k,1);
FNTest  = zeros(k,1); TNTest=zeros(k,1);

ROCCurvex = cell(k,1);
ROCCurvey = cell(k,1);
ROCCurvet = cell(k,1);
ROCCurvexTrain=cell(k,1);
ROCCurveyTrain=cell(k,1);
ROCCurvetTrain=cell(k,1);

% For storing "TRAIN-based optimal threshold"
optimalThresholds   = zeros(k,1);
optimalAccuracyVals = zeros(k,1);
optimalRecallVals   = zeros(k,1);
optimalPrecisionVals= zeros(k,1);
optimalF1Vals       = zeros(k,1);

%% Results struct
results = struct('fold',{},...
    'trainIndices',{},'testIndices',{},...
    'TPTrain',{},'FPTrain',{},'FNTrain',{},'TNTrain',{},...
    'TPTest',{}, 'FPTest',{}, 'FNTest',{}, 'TNTest',{},...
    'AccuracyTrain',{}, 'AccuracyTest',{},...
    'RecallTrain',{},   'RecallTest',{},...
    'PrecisionTrain',{},'PrecisionTest',{},...
    'F1Train',{}, 'F1Test',{},...
    'AUC',{},...
    'ROCCurvex',{},'ROCCurvey',{},'ROCCurvet',{},...
    'ROCCurvexTrain',{},'ROCCurveyTrain',{},'ROCCurvetTrain',{},...
    'YTest',{}, 'YTestPred',{}, 'YTrain',{}, 'YTrainPred',{},...
    'optimalThreshold',{},'optimalAccuracy',{},'optimalRecall',{},'optimalPrecision',{},'optimalF1',{});

%% ==================== K-fold CV Indices ==================== %%
indices = crossvalind('Kfold', numSamples, k);

fprintf('Starting %d-fold cross-validation...\n',k);

for foldID = 1:k
    fprintf('\n=== Fold %d of %d ===\n', foldID, k);

    % Split train/test
    testIdx  = (indices==foldID);
    trainIdx = ~testIdx;

    if useAllFeatures
        XTrain = X(:,:, trainIdx);
        XTest  = X(:,:, testIdx);
    else
        XTrain = X(1,:, trainIdx);
        XTest  = X(1,:, testIdx);
    end

    YTrain = goodOrBad2(trainIdx);
    YTest  = goodOrBad2(testIdx);

    fprintf('Fold %d: #Train=%d, #Test=%d\n', foldID, sum(trainIdx), sum(testIdx));

    %% (Optional) Data Augmentation or Oversampling
    XTrainAug = XTrain;
    YTrainAug = YTrain;

    %% Prepare train data => dlarray
    XTrainPerm = permute(XTrainAug,[1,3,2]);
    dlXTrain   = dlarray(XTrainPerm,'CBT');

    % Shuffle
    rp = randperm(size(dlXTrain,2));
    dlXTrain = dlXTrain(:,rp,:);
    YTrainAug= YTrainAug(rp);

    %% Define the CNN + training opts
    net = dlnetwork(layers);
    options = trainingOptions('adam',...
        'MaxEpochs',30,...
        'MiniBatchSize',64*2,...
        'Shuffle','every-epoch',...
        'Verbose',true,...
        'Plots','none',...
        'ValidationFrequency',50,...
        'InitialLearnRate',5e-4,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',1,...
        'LearnRateDropPeriod',1,...
        'Metrics',["precision","accuracy","auc","fscore","recall"],...
        'ObjectiveMetricName',"auc",...
        'OutputNetwork','last-iteration');

    %% Train
    fprintf('Training fold %d...\n', foldID);
    try
        [trainedNet, info] = trainnet(dlXTrain, YTrainAug', net, "crossentropy", options);
        fprintf('Training done.\n');
    catch ME
        warning('Fold %d: error training => skip.\n%s', foldID, ME.message);
        accuracyScores(foldID)=NaN;
        continue;
    end

    %% Evaluate on TRAIN
    fprintf('Predicting on TRAIN for fold %d...\n', foldID);
    YPredTrain = predict(trainedNet, dlXTrain);
    YPredTrain = extractdata(YPredTrain); % 2 x #train
    [~, trainMaxIdx] = max(YPredTrain,[],1);

    trainCats = categories(YTrainAug); % e.g. {'reject','accept'}
    YPredTrainLabels = trainCats(trainMaxIdx);

    % Compute AUC on train
    [fprTrain, tprTrain, thrTrain, aucTrain] = perfcurve(YTrainAug, YPredTrain(1,:)', 'reject');
    aucScoresTrain(foldID) = aucTrain;
    ROCCurvexTrain{foldID} = fprTrain;
    ROCCurveyTrain{foldID} = tprTrain;
    ROCCurvetTrain{foldID} = thrTrain;

    % ConfMat => recall, precision, F1, accuracy
    confMatTrain = confusionmat(YTrainAug, categorical(YPredTrainLabels'));
    [recT, precT, f1T, TPTr, FPTr, FNTr, TNTr] = computeMetrics(confMatTrain);
    accuracyTrainFold = mean(YPredTrainLabels==YTrainAug');

    trainAccuracyScores(foldID)= accuracyTrainFold;
    trainRecallScores(foldID)  = recT;
    trainPrecisionScores(foldID)= precT;
    trainF1Scores(foldID)      = f1T;

    %% >>>> Find "optimal threshold" from TRAIN ROC <<<<
    dist2CornerTrain = sqrt( (fprTrain - 0).^2 + (tprTrain - 1).^2 );
    [~, bestIdxTrain] = min(dist2CornerTrain);
    trainOptThreshold = thrTrain(bestIdxTrain);

    %% Evaluate on TEST (default approach)
    XTestPerm = permute(XTest,[1,3,2]);
    dlXTest   = dlarray(XTestPerm, 'CBT');
    YPredTest = predict(trainedNet, dlXTest);
    YPredTest = extractdata(YPredTest); % 2 x #test

    % default labels => max
    [~, testMaxIdx] = max(YPredTest,[],1);
    testCats = categories(YTest);
    YPredTestLabels = testCats(testMaxIdx);

    % default confusion + metrics
    [fprTest, tprTest, thrTest, aucTest] = perfcurve(YTest, YPredTest(1,:)', 'reject');
    aucScores(foldID) = aucTest;
    confMatTest = confusionmat(YTest, categorical(YPredTestLabels'));
    [recTest, precTest, f1Test, TPTe, FPTe, FNTe, TNTe] = computeMetrics(confMatTest);
    accuracyTestFold = mean(YPredTestLabels==YTest');

    accuracyScores(foldID)   = accuracyTestFold;
    recallScores(foldID)     = recTest;
    precisionScores(foldID)  = precTest;
    f1Scores(foldID)         = f1Test;

    ROCCurvex{foldID} = fprTest;
    ROCCurvey{foldID} = tprTest;
    ROCCurvet{foldID} = thrTest;

    fprintf('\nFold %d => Default threshold results:\n', foldID);
    fprintf('Test Accuracy=%.2f%%, AUC=%.3f, R=%.3f, P=%.3f, F1=%.3f\n',...
        100*accuracyTestFold, aucTest, recTest, precTest, f1Test);

    %% Evaluate on TEST with train-based "optimal threshold"
    probRejectTest = YPredTest(1,:); % row1 => prob(reject)
    isRejectOpt = (probRejectTest >= trainOptThreshold);
    strOptLabels= strings(size(isRejectOpt));
    strOptLabels(isRejectOpt)="reject";
    strOptLabels(~isRejectOpt)="accept";
    catOptLabels= categorical(strOptLabels,{'reject','accept'});

    confMatOpt = confusionmat(YTest,catOptLabels);
    [optRec, optPrec, optF1, optTP, optFP, optFN, optTN] = computeMetrics(confMatOpt);
    optAcc = mean(catOptLabels==YTest);

    optimalThresholds(foldID)       = trainOptThreshold;
    optimalAccuracyVals(foldID)     = optAcc;
    optimalRecallVals(foldID)       = optRec;
    optimalPrecisionVals(foldID)    = optPrec;
    optimalF1Vals(foldID)           = optF1;

    fprintf('Fold %d => Train-based optimum threshold=%.4f\n', foldID, trainOptThreshold);
    fprintf('Test(OptThresh): Acc=%.3f, Rec=%.3f, Prec=%.3f, F1=%.3f\n',...
        optAcc, optRec, optPrec, optF1);

    %% CSV of misclassifications using "optimal" threshold
    if haveRow && haveCol && haveIndices
        testAbsInd = find(testIdx);
        YTestCell  = cellstr(YTest);
        YPredOpt   = cellstr(strOptLabels');
        mismatchMask = ~strcmp(string(YPredOpt), string(YTestCell)');
        badIdx       = testAbsInd(mismatchMask);

        if isempty(badIdx)
            fprintf('No misclassifications (train-based threshold) in fold %d.\n', foldID);
        else
            fileRows  = data.fileRow(badIdx);
            fileCols  = data.fileCol(badIdx);
            fileNames = data.fileIndices(badIdx);

            actualLab = YTestCell(mismatchMask);
            predLab   = YPredOpt(mismatchMask);

            Twrong = table(fileRows(:), fileCols(:), fileNames(:),...
                actualLab(:), predLab(:),...
                'VariableNames',{'fileRow','fileCol','fileName','actualLabel','predictedLabel'});

            csvName = sprintf('%s_fold%d_wrongPredictions_trainOpt.csv',...
                erase(saveName,'.mat'), foldID);
            fprintf('Writing misclassifications (train-based threshold) to "%s"...\n', csvName);
            writetable(Twrong, csvName);
        end
    else
        fprintf('[Skipping CSV misclassification because fileRow/Col/Indices not found.]\n');
    end

    %% Store results
    results(foldID).fold = foldID;
    results(foldID).trainIndices = find(trainIdx);
    results(foldID).testIndices  = find(testIdx);

    results(foldID).TPTrain=TPTr;  results(foldID).FPTrain=FPTr;
    results(foldID).FNTrain=FNTr;  results(foldID).TNTrain=TNTr;
    results(foldID).TPTest =TPTe;  results(foldID).FPTest=FPTe;
    results(foldID).FNTest =FNTe;  results(foldID).TNTest=TNTe;

    results(foldID).AccuracyTrain   = accuracyTrainFold;
    results(foldID).AccuracyTest    = accuracyTestFold;
    results(foldID).RecallTrain     = recT;
    results(foldID).RecallTest      = recTest;
    results(foldID).PrecisionTrain  = precT;
    results(foldID).PrecisionTest   = precTest;
    results(foldID).F1Train         = f1T;
    results(foldID).F1Test          = f1Test;
    results(foldID).AUC             = aucTest;  % test set AUC
    results(foldID).ROCCurveX       = fprTest;
    results(foldID).ROCCurveY       = tprTest;
    results(foldID).ROCCurveT       = thrTest;

    results(foldID).ROCCurvexTrain  = fprTrain;
    results(foldID).ROCCurveyTrain  = tprTrain;
    results(foldID).ROCCurvetTrain  = thrTrain;
    results(foldID).YTest           = YTest;
    results(foldID).YTestPred       = YPredTest(1,:)';
    results(foldID).YTrain          = YTrainAug;
    results(foldID).YTrainPred      = YPredTrain(1,:);

    % "train-based optimal" approach
    results(foldID).optimalThreshold= trainOptThreshold;
    results(foldID).optimalAccuracy = optAcc;
    results(foldID).optimalRecall   = optRec;
    results(foldID).optimalPrecision= optPrec;
    results(foldID).optimalF1       = optF1;
end

%% Summaries
validFolds = ~isnan(accuracyScores);
meanAccuracy      = mean(accuracyScores(validFolds));
meanTrainAccuracy = mean(trainAccuracyScores(validFolds));
meanRecall        = mean(recallScores(validFolds));
meanPrecision     = mean(precisionScores(validFolds));
meanF1Score       = mean(f1Scores(validFolds));
meanAUC           = mean(aucScores(validFolds));
meanTrainRecall   = mean(trainRecallScores(validFolds));
meanTrainPrecision= mean(trainPrecisionScores(validFolds));
meanTrainF1Score  = mean(trainF1Scores(validFolds));
meanTrainAUC      = mean(aucScoresTrain(validFolds));

meanOptThreshold  = mean(optimalThresholds(validFolds));
meanOptAccuracy   = mean(optimalAccuracyVals(validFolds));
meanOptRecall     = mean(optimalRecallVals(validFolds));
meanOptPrecision  = mean(optimalPrecisionVals(validFolds));
meanOptF1         = mean(optimalF1Vals(validFolds));

fprintf('\n=== Overall (Default) Metrics ===\n');
fprintf('Mean Accuracy (Test)=%.2f%%\n', meanAccuracy*100);
fprintf('Mean AUC (Test)=%.3f\n', meanAUC);
fprintf('Mean Recall (Test)=%.3f\n', meanRecall);
fprintf('Mean Precision (Test)=%.3f\n', meanPrecision);
fprintf('Mean F1 (Test)=%.3f\n', meanF1Score);
fprintf('Mean Accuracy (Train)=%.2f%%\n', meanTrainAccuracy*100);
fprintf('Mean AUC (Train)=%.3f\n', meanTrainAUC);
fprintf('Mean Recall (Train)=%.3f\n', meanTrainRecall);
fprintf('Mean Precision (Train)=%.3f\n', meanTrainPrecision);
fprintf('Mean F1 (Train)=%.3f\n', meanTrainF1Score);

fprintf('\n=== Train-Based Optimal Threshold Approach ===\n');
fprintf('Avg Optimal Threshold=%.4f\n', meanOptThreshold);
fprintf('Avg Accuracy=%.3f\n', meanOptAccuracy);
fprintf('Avg Recall=%.3f\n', meanOptRecall);
fprintf('Avg Prec=%.3f\n', meanOptPrecision);
fprintf('Avg F1=%.3f\n', meanOptF1);

%% Plot ROC
close all
fprintf('Plotting test ROC curves (default approach) for each fold...\n');
figure('Name','Test ROC (Default) for All Folds');
hold on; grid on;
colors = copper(k);

for i = 1:k
    if ~isempty(ROCCurvex{i})
        plot(ROCCurvex{i}, ROCCurvey{i},'DisplayName',...
            sprintf('Fold %d (AUC=%.2f)', i, aucScores(i)), ...
            'Color',colors(i,:));
    end
end
xlabel('FPR'); ylabel('TPR');
%title('Test ROC Curves (Default) for Each Fold');
legend('Location','best');
hold off;

fprintf('Plotting training ROC curves (default approach) for each fold...\n');
figure('Name','Training ROC (Default) for All Folds');
hold on; grid on;
for i = 1:k
    if ~isempty(ROCCurvexTrain{i})
        plot(ROCCurvexTrain{i}, ROCCurveyTrain{i}, 'DisplayName', ...
            sprintf('Fold %d (AUC=%.2f)', i, aucScoresTrain(i)), ...
            'Color',colors(i,:));
    end
end
xlabel('FPR'); 
ylabel('TPR');
%title('Training ROC Curves (Default) for Each Fold');
legend('Location','best');
hold off;

%% Save
fprintf('Saving final results to "%s"\n', saveName);
save(saveName, ...
    'results',...
    'accuracyScores','trainAccuracyScores',...
    'recallScores','precisionScores','f1Scores','aucScores',...
    'trainRecallScores','trainPrecisionScores','trainF1Scores','aucScoresTrain',...
    'meanAccuracy','meanTrainAccuracy','meanRecall','meanPrecision','meanF1Score','meanAUC',...
    'meanTrainRecall','meanTrainPrecision','meanTrainF1Score','meanTrainAUC',...
    'optimalThresholds','optimalAccuracyVals','optimalRecallVals','optimalPrecisionVals','optimalF1Vals',...
    'meanOptThreshold','meanOptAccuracy','meanOptRecall','meanOptPrecision','meanOptF1',...
    'k', "ROCCurvet","ROCCurvetTrain", "optimalThresholds","ROCCurvex", ...
    "ROCCurvey","ROCCurvexTrain", "ROCCurveyTrain" );

fprintf('Done.\n');

end % function


%% --------------------------------------------------------------------------
function [recall, precision, f1, TP, FP, FN, TN] = computeMetrics(confMat)
% confMat=[TP,FN; FP,TN]
TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

recall    = TP/(TP+FN);
precision = TP/(TP+FP);
f1        = 2*(precision*recall)/(precision+recall);
end
