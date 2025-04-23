clear
clc
close all
%% EvrenFineTuneScript.m
% Fine‐tune pretrained COBRA regression model on Evren annotations
% for a range of training set sizes (0:100:500) with 5 bootstrap replicates,
% evaluating MAE in nm on a fixed hold‐out half.

% ------------------------ USER PARAMETERS ------------------------- %
oldModelFile       = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\trainedRegressionModels\pooling_after_bilstm_2conv_relu.mat';
preprocDataFile    = 'C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_Evren_annotated.mat';
saveName           = 'EvrenFineTuneResults.mat';
useAugmentation    = false;    % set false to disable augmentation

curveTrainSizes    = 0:200:1000;  % numbers of curves for fine‐tuning
nBootstraps        = 10;          % bootstrap replicates per size
seedNumber         = 1337;       % for reproducibility

rng(seedNumber);

% ------------------------ LOAD PRETRAINED MODEL ------------------------ %
fprintf('Loading pretrained model from:\n  %s\n', oldModelFile);
mdl = load(oldModelFile);
if isfield(mdl,'net')
    net0 = mdl.net;
elseif isfield(mdl,'trainedNet')
    net0 = mdl.trainedNet;
else
    error('No network variable (net or trainedNet) found in %s', oldModelFile);
end

% ------------------------ LOAD PREPROCESSED DATA ------------------------ %
fprintf('Loading Evren‐annotated data from:\n  %s\n', preprocDataFile);
D = load(preprocDataFile);
X        = D.X;                 % [C x T x N]
Y        = D.Y;                 % [N x 1]
maxExt   = D.maxExtValues';     % [N x 1]
minExt   = D.minExtValues';     % [N x 1]
numSamples = size(X,3);

% split half for hold‐out
nHold = floor(numSamples/2);
perm  = randperm(numSamples);
holdIdx = perm(1:nHold);
poolIdx = perm(nHold+1:end);

XHold = X(:,:,holdIdx);
YHold = Y(holdIdx);

fprintf('Hold‐out curves: %d, Training pool: %d\n', nHold, numel(poolIdx));

% ------------------------ PREALLOCATE RESULTS ------------------------ %
totalRuns = numel(curveTrainSizes)*nBootstraps;
results(totalRuns) = struct(...
    'curveTrainSize',[],...
    'bootstrapIndex',[],...
    'timeToTrain',[],...
    'holdoutMAE_nm',[]);

runCounter = 1;

% ------------------------ FINE‐TUNING LOOP ------------------------ %
for sz = curveTrainSizes
    fprintf('=== Training size = %d ===\n', sz);
    for b = 1:nBootstraps
        fprintf('  Bootstrap %d/%d...\n', b, nBootstraps);

        if sz > 0
            if sz > numel(poolIdx)
                error('Requested %d training curves but only %d available.', sz, numel(poolIdx));
            end
            selIdx = randsample(poolIdx, sz);
            XTrain = X(:,:,selIdx);
            YTrain = Y(selIdx);

            if useAugmentation
                [XTrain, YTrain] = augmentData(XTrain, YTrain, 100);
            end

            tic;
            net = trainModelCoreFineTuning(net0, XTrain, YTrain);
            tTrain = toc;
        else
            net    = net0;
            tTrain = 0;
        end

        % prepare hold‐out for prediction
        dlX = dlarray(permute(XHold,[1 3 2]), 'CBT');
        YPred = extractdata(predict(net, dlX))';

        % convert to nm
        extRange = (maxExt(holdIdx) - minExt(holdIdx));
        pred_nm  = YPred .* extRange + minExt(holdIdx);
        true_nm  = YHold .* extRange + minExt(holdIdx);
        mae_nm   = mean(abs(pred_nm - true_nm));

        % store
        results(runCounter).curveTrainSize = sz;
        results(runCounter).bootstrapIndex = b;
        results(runCounter).timeToTrain     = tTrain;
        results(runCounter).holdoutMAE_nm   = mae_nm;
        runCounter = runCounter + 1;
    end
end

% ------------------------ SAVE RESULTS ------------------------ %
fprintf('Saving results to:\n  %s\n', saveName);
save(saveName, 'results', 'curveTrainSizes', 'nBootstraps', 'holdIdx', 'poolIdx', 'seedNumber');
fprintf('Done.\n');
