function metricsOut = testTrainedModelOnDataset_classification( ...
    trainedNet, XtestAll, YtestAll, threshold, saveTestResults, saveFileName)
% TESTTRAINEDMODELONDATASET_CLASSIFICATION
%   Takes a trained dlnetwork + test data (XtestAll, YtestAll) + threshold,
%   computes classification metrics, returns a struct 'metricsOut'.
%   "reject"=0 is positive class => the net's output #1 is probOfReject.
%
% Inputs:
%   trainedNet    : dlnetwork (2 outputs: #1=reject, #2=accept)
%   XtestAll      : [features x seqLength x #samples]
%   YtestAll      : numeric array (#samples) => 0=reject,1=accept
%   threshold     : numeric, e.g. 0.5 => if probReject >= threshold => predict=reject
%                   or the string 'optimal' => pick threshold from ROC
%   saveTestResults : boolean
%   saveFileName     : path if saving
%
% Output: metricsOut => fields: accuracy, auc, recall, precision, f1, etc.
%         plus .usedThreshold => the numeric threshold we used
%

    if nargin<4 || isempty(threshold)
        threshold = 0.5;
    end

    metricsOut = struct();
    if isempty(trainedNet) || isempty(XtestAll) || isempty(YtestAll)
        warning('Empty net or test data => return NaNs');
        metricsOut.accuracy=NaN; 
        metricsOut.auc=NaN;
        metricsOut.recall=NaN; 
        metricsOut.precision=NaN; 
        metricsOut.f1=NaN;
        metricsOut.usedThreshold=NaN;
        return;
    end

    nTest = size(XtestAll,3);
    if numel(YtestAll)~=nTest
        warning('Mismatch in XtestAll vs. YtestAll => NaNs');
        metricsOut.accuracy=NaN; 
        metricsOut.auc=NaN;
        metricsOut.recall=NaN; 
        metricsOut.precision=NaN; 
        metricsOut.f1=NaN;
        metricsOut.usedThreshold=NaN;
        return;
    end

    % Convert YtestAll => categorical
    strTest = strings(nTest,1);
    strTest(YtestAll==0)="reject";
    strTest(YtestAll==1)="accept";
    YcatTest = categorical(strTest, {'reject','accept'});

    % permute
    XtestPerm = permute(XtestAll,[1,3,2]);
    dlXtest   = dlarray(XtestPerm,'CBT');

    % predict
    scores = predict(trainedNet, dlXtest); % 2 x nTest
    scores = extractdata(scores);
    probReject = scores(1,:);  % row #1 => prob of 'reject'

    % compute AUC => reject=pos
    [fpr, tpr, thr, aucVal] = perfcurve(YcatTest, probReject', 'reject');

    if ischar(threshold) && strcmpi(threshold,'optimal')
        % 1) We find the threshold that is closest to top-left corner = (FPR=0, TPR=1)
        dist2Corner = sqrt( (fpr - 0).^2 + (tpr - 1).^2 );
        [~, bestIdx] = min(dist2Corner);
        finalThreshold = thr(bestIdx);

        fprintf('Using "optimal" threshold = %.4f (min distance=%.4f)\n',...
            finalThreshold, dist2Corner(bestIdx));
    else
        % numeric threshold
        finalThreshold = threshold;
        fprintf('Using numeric threshold=%.4f\n', finalThreshold);
    end

    % apply finalThreshold
    isReject = (probReject >= finalThreshold);
    strPred  = strings(nTest,1);
    strPred(isReject)="reject";
    strPred(~isReject)="accept";
    YcatPred = categorical(strPred,{'reject','accept'});

    % confusion matrix
    confMat = confusionmat(YcatTest, YcatPred, 'Order',{'reject','accept'});
    TP = confMat(1,1);
    FN = confMat(1,2);
    FP = confMat(2,1);
    TN = confMat(2,2);

    accuracy = (TP+TN)/sum(confMat(:));
    if (TP+FN)==0
        recall=NaN;
    else
        recall=TP/(TP+FN);
    end

    if (TP+FP)==0
        precision=NaN;
    else
        precision=TP/(TP+FP);
    end

    if isnan(recall)||isnan(precision)||(recall+precision)==0
        f1v=NaN;
    else
        f1v= 2*(precision*recall)/(precision+recall);
    end

    metricsOut.accuracy  = accuracy;
    metricsOut.auc       = aucVal;
    metricsOut.recall    = recall;
    metricsOut.precision = precision;
    metricsOut.f1        = f1v;
    metricsOut.TP        = TP; 
    metricsOut.FN        = FN; 
    metricsOut.FP        = FP; 
    metricsOut.TN        = TN;
    metricsOut.usedThreshold = finalThreshold; % store the numeric threshold used

    fprintf('Tested on %d curves. threshold=%.2f => Acc=%.2f%%, AUC=%.3f, R=%.3f, P=%.3f, F1=%.3f\n',...
        nTest, finalThreshold, 100*accuracy, aucVal, recall, precision, f1v);

    if saveTestResults
        if nargin<6 || isempty(saveFileName)
            saveFileName = ['testResults_classif_' datestr(now,'yyyymmdd_HHMM') '.mat'];
        end
        save(saveFileName,'-struct','metricsOut');
    end
end
