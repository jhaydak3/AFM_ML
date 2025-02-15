%% SCRIPT: trainTestByQuartilesClassification_NoLeftover.m
%
% - We split data into 4 bins by the quartiles of 'modulusHertz' => these bins
%   represent 4 "domains."
% - For each domain bin i:
%     (a) 5-fold cross-validation (i->i) => measure self-domain metrics (accuracy, AUC, recall)
%     (b) Train a single "all-data" model on bin i => test it on bin j => cross-domain
%         => store cross-bin metrics in [i,j].
%


clear; clc; close all;

%% 1) Load data (for classification, but using modulusHertz for domain partition)
dataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\classification_processed_files\processed_features_for_classification_All.mat";
addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions");

S = load(dataFile, 'X','goodOrBad','modulusHertz');
if ~isfield(S,'X') || ~isfield(S,'goodOrBad') || ~isfield(S,'modulusHertz')
    error('Missing X, goodOrBad, or modulusHertz in "%s".', dataFile);
end
X            = S.X;            % [features x seqLen x N]
labels       = S.goodOrBad;    % 0/1 for classification
modulusHertz = S.modulusHertz; % numeric vector length N
N = size(X,3);
fprintf('Loaded %d curves from "%s".\n', N, dataFile);

%% 2) Assign 4 bins by quartiles of modulusHertz => domain partition
K = 4;
edges = quantile(modulusHertz, [0 0.25 0.5 0.75 1.0]);
binIndex = zeros(N,1);
for i=1:N
    val = modulusHertz(i);
    if val < edges(2)
        binIndex(i)=1;
    elseif val < edges(3)
        binIndex(i)=2;
    elseif val < edges(4)
        binIndex(i)=3;
    else
        binIndex(i)=4;
    end
end

%% 3) We'll store classification metrics in KxK
accuracyMatrix = nan(K,K);
aucMatrix      = nan(K,K);
recallMatrix   = nan(K,K);  % you can add more (precision, F1, etc.) if desired

%% 4) For cross-domain usage, we need 1 "all-data" model trained on bin i
%    plus a self-test (5-fold CV) for i->i
modelsAllData = cell(K,1);

%% 5) Define your classification CNN architecture
nFeatures      = size(X,1);
sequenceLength = size(X,2);
layers = CNN_custom_pooling_after_lstm_2conv_relu_classification( ...
    nFeatures, sequenceLength, 2); 
% ^ must output 2 classes => ["reject","accept"]

rng(1337,'twister');  % reproducible

%% 6) For each bin i:
for iBin = 1:K
    fprintf('\n=== DOMAIN BIN %d ===\n', iBin);
    binMask = (binIndex==iBin);
    numCurvesBin = sum(binMask);
    fprintf('Bin %d has %d curves.\n', iBin, numCurvesBin);

    if numCurvesBin < 2
        fprintf('Not enough curves => skip this bin.\n');
        continue;
    end

    % (A) 5-fold self-test => i->i
    fprintf('  -> Doing 5-fold crossvalidation for self-domain (i->i)...\n');
    % Pass 'S' so that evaluateClassificationSet can access it if needed
    metricsCV = doKfoldClassificationInBin(X, labels, binMask, layers, S);
    accuracyMatrix(iBin, iBin) = metricsCV.accuracy;
    aucMatrix(iBin, iBin)      = metricsCV.auc;
    recallMatrix(iBin, iBin)   = metricsCV.recall;

    % (B) Train a single model on ALL curves in bin i => cross-domain usage
    fprintf('  -> Training single "all-data" model for cross-bin usage...\n');
    Xbin = X(:,:, binMask);
    Lbin = labels(binMask);

    if numCurvesBin<2
        fprintf('    skip => not enough data.\n');
        modelsAllData{iBin} = [];
    else
        netTrained = trainAllDataClassification(Xbin, Lbin, layers);
        modelsAllData{iBin} = netTrained;
    end
end

%% 7) Cross-domain tests: i-> j, for i != j
for iBin = 1:K
    modelCross = modelsAllData{iBin};
    if isempty(modelCross)
        continue;
    end

    for jBin = 1:K
        if jBin == iBin
            continue;  % we already did i->i (via k-fold)
        end

        fprintf('\nCross test: model from bin %d => bin %d\n', iBin, jBin);
        maskTest = (binIndex == jBin);
        nTest    = sum(maskTest);
        if nTest<1
            fprintf('  No curves => skip.\n');
            continue;
        end

        XtestAll = X(:,:, maskTest);
        LtestAll = labels(maskTest);

        % Evaluate classification
        % Also pass 'S' if you want to do advanced calculations in evaluateClassificationSet
        metricsCross = evaluateClassificationSet(modelCross, XtestAll, LtestAll, S);

        accuracyMatrix(iBin,jBin) = metricsCross.accuracy;
        aucMatrix(iBin,jBin)      = metricsCross.auc;
        recallMatrix(iBin,jBin)   = metricsCross.recall;

        fprintf('  #test=%d => Acc=%.2f%%, AUC=%.3f, Recall=%.3f\n', ...
            nTest, 100*metricsCross.accuracy, metricsCross.auc, metricsCross.recall);
    end
end

%% 8) Visualize
figure('Name','Accuracy');
heatmap(1:K, 1:K, accuracyMatrix, 'Title','Accuracy (train -> test)');
xlabel('Test Hertzian Modulus Quartile'); ylabel('Train Hertzian Modulus Quartile');

figure('Name','AUC');
xStr = {'Q_1','Q_2','Q_3','Q_4'};
h = heatmap(xStr, xStr, round(aucMatrix,2));
%title('AUC'
xlabel('Test Hertzian Modulus Quartile'); ylabel('Train Hertzian Modulus Quartile');
colormap(magma)

figure('Name','Recall');
heatmap(1:K,1:K, recallMatrix, 'Title','Recall (train -> test)');
xlabel('Test Hertzian Modulus Quartile'); ylabel('Train Hertzian Modulus Quartile');

save('quartileDomainClassification.mat','K','accuracyMatrix','aucMatrix','recallMatrix');
fprintf('\nQuartile-based domain classification script completed.\n');



%% ------------------------------------------------------------------
%% Local subfunction: doKfoldClassificationInBin => i-> i
%%    5-fold CV on data from bin i only (domain i).
%%    We measure self-domain performance => store in [i,i].
function metricsCV = doKfoldClassificationInBin(X, labels, binMask, layers, S)
    % Extract bin i data
    Xbin = X(:,:,binMask);
    Lbin = labels(binMask);
    Nbin = sum(binMask);

    metricsCV = struct('accuracy',NaN,'auc',NaN,'recall',NaN);
    if Nbin < 2
        return;
    end

    k = 5;
    if Nbin < k
        warning('Bin has fewer than %d => skipping => NaN.', k);
        return;
    end

    indices = crossvalind('Kfold', Nbin, k);
    accAll   = nan(k,1);
    aucAll   = nan(k,1);
    recAll   = nan(k,1);

    for foldID = 1:k
        testMask  = (indices == foldID);
        trainMask = ~testMask;

        Xtrain = Xbin(:,:, trainMask);
        Ltrain = Lbin(trainMask);

        Xtest  = Xbin(:,:, testMask);
        Ltest  = Lbin(testMask);

        % Train
        netFold = trainAllDataClassification(Xtrain, Ltrain, layers);
        if isempty(netFold)
            continue;
        end

        % Evaluate (pass S if you want advanced logic in evaluateClassificationSet)
        outFold = evaluateClassificationSet(netFold, Xtest, Ltest, S);

        accAll(foldID)  = outFold.accuracy;
        aucAll(foldID)  = outFold.auc;
        recAll(foldID)  = outFold.recall;
    end

    metricsCV.accuracy = mean(accAll,'omitnan');
    metricsCV.auc      = mean(aucAll,'omitnan');
    metricsCV.recall   = mean(recAll,'omitnan');
end


%% ------------------------------------------------------------------
%% trainAllDataClassification => trains a single classification model
%%   on the entire domain bin (no leftover).
function netTrained = trainAllDataClassification(Xbin, Lbin, layers)
    % Xbin: [features x seqLen x N]
    % Lbin: [N x 1] or [1 x N], 0/1
    netTrained = [];

    Nbin = size(Xbin,3);
    if Nbin < 2
        return;
    end

    % Convert numeric => categorical
    strLabels = strings(Nbin,1);
    strLabels(Lbin==0) = "reject";
    strLabels(Lbin==1) = "accept";
    Ycat = categorical(strLabels, ["reject","accept"]);

    net = dlnetwork(layers);

    % Convert X => 'CBT'
    Xp = permute(Xbin,[1 3 2]);  % [features x N x seqLen]
    dlX= dlarray(Xp,'CBT');

    % Shuffle
    rp = randperm(Nbin);
    dlX   = dlX(:, rp, :);
    Ycat  = Ycat(rp);

    opts = trainingOptions('adam', ...
        'MaxEpochs',30, ...
        'MiniBatchSize',64*.5, ...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',5e-4, ...
        'Plots','none', ...
        'Verbose',true);

    try
        [netTrained, info] = trainnet(dlX, Ycat, net, "crossentropy", opts);
    catch ME
        warning('Error training => returning empty: %s', ME.message);
        netTrained = [];
    end
end


%% ------------------------------------------------------------------
%% evaluateClassificationSet => predict => compute accuracy, AUC, recall
function metricsOut = evaluateClassificationSet(net, Xtest, Ltest, S)
    % net: dlnetwork for classification
    % Xtest: [features x seqLen x Ntest]
    % Ltest: 0/1 array
    % S: optional struct if you want to do advanced logic (like measure
    %    how performance depends on raw data, or compute "modulus" metrics).
    %
    if nargin<4, S=[]; end  % default empty if not provided

    metricsOut = struct('accuracy',NaN,'auc',NaN,'recall',NaN);
    if isempty(net) || isempty(Xtest) || isempty(Ltest)
        return;
    end

    Ntest = size(Xtest,3);

    % Convert numeric => "reject"/"accept"
    strTest = strings(Ntest,1);
    strTest(Ltest==0) = "reject";
    strTest(Ltest==1) = "accept";
    YcatTest = categorical(strTest, ["reject","accept"]);

    % Predict
    Xp   = permute(Xtest,[1 3 2]); % => [features x N x seqLen]
    dlXt = dlarray(Xp,'CBT');
    out  = predict(net, dlXt);  % => [2 x Ntest]
    out2 = gather(extractdata(out))'; % Nx2: [probReject, probAccept]

    % (A) AUC (threshold-free)
    [~,~,~,aucVal] = perfcurve(YcatTest, out2(:,1), "reject"); 

    % (B) Fix threshold => e.g. 0.5 for accept vs. reject
    predReject = (out2(:,1) >= 0.5);
    strPred = repmat("accept",Ntest,1);
    strPred(predReject) = "reject";
    catPred = categorical(strPred, ["reject","accept"]);

    % confusion matrix => C = [TP,FN; FP,TN] if categories=["reject","accept"]
    C = confusionmat(YcatTest, catPred);
    TP = C(1,1); FN = C(1,2);
    FP = C(2,1); TN = C(2,2);

    accuracy = (TP+TN)/(TP+TN+FP+FN+eps);
    recall   = TP/(TP+FN+eps);

    metricsOut.accuracy = accuracy;
    metricsOut.auc      = aucVal;
    metricsOut.recall   = recall;

    % (C) If you want to do something with S here (like measuring
    % performance across the test domain’s raw data, etc.), add logic below.
    % e.g. if ~isempty(S), do advanced stuff ...
end
