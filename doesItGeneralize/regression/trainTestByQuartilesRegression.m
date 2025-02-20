%% SCRIPT: trainTestByQuartilesRegression_NoLeftover.m
%
% - We split data into 4 bins by the quartiles of 'modulusHertz'.
% - For each bin i:
%     (a) We do a 5-fold cross-validation within bin i => store the average
%         self-test result in [i,i].
%     (b) We train one "all data" model on bin i => test it on bin j (j != i)
%         => store cross-bin results in [i,j].
%


clear; clc; close all;

%% 1) Load data (for regression)
dataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat";
addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\helperFunctions");

indentationDepthNm = 500;

S = load(dataFile, 'X','Y','modulusHertz','goodOrBad',...
    'rawExt','rawDefl','minExtValues','maxExtValues','spring_constant',...
    'R','v','th','b');
if ~isfield(S,'X') || ~isfield(S,'Y') || ~isfield(S,'modulusHertz')
    error('Missing X, Y, or modulusHertz in "%s".', dataFile);
end
X = S.X;                 % [features x seqLen x N]
Y = S.Y;                 % Nx1 => normalized CP
modulusHertz = S.modulusHertz; 
goodOrBad = S.goodOrBad;
N = size(X,3);
fprintf('Loaded %d curves from "%s".\n', N, dataFile);


%% 2) Assign 4 bins by quartiles
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

%% 3) We'll store final results in KxK
% e.g. MAE in normalized CP, MAE in nm, etc.
maeNormMatrix   = nan(K,K);
maeNmMatrix     = nan(K,K);
hertzMapeMatrix = nan(K,K);
mod500MapeMatrix= nan(K,K);

%% 4) For cross-Quartile testing, we need a model trained on "all" data in bin i
modelsAllData = cell(K,1);  % each cell => net trained on entire bin i

% 4a) We'll define your CNN architecture for regression
nFeatures      = size(X,1);
sequenceLength = size(X,2);
%layers = CNN_custom_pooling_after_lstm_relu_regression(nFeatures, sequenceLength);
layers = CNN_custom_pooling_after_bilstm_2conv_relu(nFeatures, sequenceLength, 7);

% 4b) trainingOptions
opts = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'MiniBatchSize',64*.5, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-4, ...
    'Plots','none', ...
    'Verbose',true);

rng(1337,'twister');  % reproducible

%% 5) For each bin i:
for iBin = 1:K
    fprintf('\n=== BIN %d ===\n', iBin);
    binMask = (binIndex==iBin);
    numCurvesBin = sum(binMask);
    fprintf('Bin %d has %d curves.\n', iBin, numCurvesBin);

    if numCurvesBin < 2
        fprintf('Not enough curves => skip.\n');
        continue;
    end

    % (A) K-fold self-test => i->i
    fprintf('  -> Doing 5-fold crossvalidation for self-test...\n');
    metricsCV = doKfoldRegressionInBin(X, Y, goodOrBad, binMask, layers, opts, S);
    maeNormMatrix(iBin,iBin)   = metricsCV.maeNorm;
    maeNmMatrix(iBin,iBin)     = metricsCV.maeNm;
    hertzMapeMatrix(iBin,iBin) = metricsCV.hertzMAPE;
    mod500MapeMatrix(iBin,iBin)= metricsCV.mod500MAPE;

    % (B) Train a single model on all curves in bin i for cross-bin usage
    fprintf('  -> Training single "all-data" model for cross-bin usage...\n');
    Xbin = X(:,:, binMask);
    Ybin = Y(binMask);

    if numCurvesBin<2
        fprintf('    skip => not enough data.\n');
        modelsAllData{iBin} = [];
    else
        % Build net
        net = dlnetwork(layers);

        % Convert X => 'CBT'
        Xp = permute(Xbin, [1 3 2]);
        dlX= dlarray(Xp, 'CBT');

        % Shuffle
        rp = randperm(numCurvesBin);
        dlX = dlX(:, rp, :);
        Ybin= Ybin(rp);

        try
            [trainedNet, info] = trainnet(dlX, Ybin, net, "mae", opts);
            modelsAllData{iBin} = trainedNet;
        catch ME
            warning('Error training bin %d => cross model empty.\n%s', iBin, ME.message);
            modelsAllData{iBin} = [];
        end
    end
end

%% 6) Cross-Quartile Tests: i-> j (i != j)
for iBin = 1:K
    netCross = modelsAllData{iBin};
    if isempty(netCross)
        continue;
    end

    for jBin = 1:K
        if jBin == iBin, continue; end  % we already did i->i as K-fold

        fprintf('\nCross test: model from bin %d => bin %d\n', iBin, jBin);
        maskTest = (binIndex == jBin);
        XtestAll = X(:,:, maskTest);
        YtestAll = Y(maskTest);
        nTest    = sum(maskTest);

        if nTest<1
            fprintf('  No curves => skip.\n');
            continue;
        end

        % Evaluate
        % We'll call a helper that calculates MAE norm, nm, plus modulus MAPE if desired
        metricsCross = evaluateRegressionSet(netCross, XtestAll, YtestAll, ...
            S, find(maskTest), indentationDepthNm);  % pass 'S' with rawExt etc., plus testInds

        maeNormMatrix(iBin,jBin)   = metricsCross.maeNorm;
        maeNmMatrix(iBin,jBin)     = metricsCross.maeNm;
        hertzMapeMatrix(iBin,jBin) = metricsCross.hertzMAPE;
        mod500MapeMatrix(iBin,jBin)= metricsCross.mod500MAPE;

        fprintf('  #test=%d => maeNorm=%.3f, maeNm=%.1f, hertzMAPE=%.2f%%, 500nmMAPE=%.2f%%\n', ...
            nTest, metricsCross.maeNorm, metricsCross.maeNm, ...
            metricsCross.hertzMAPE, metricsCross.mod500MAPE);
    end
end

%% 7) Visualize


fontSize = 18;
fontSizeLegend = 13;
fontFamily = 'Arial';
cellLabelColorStr = 'auto';



xStr = {'Quartile 1','Quartile 2','Quartile 3','Quartile 4'};

figure('Name','MAE CP', ...
    'Units','inches', 'Position',[1 1 5 5]*1.3);
set(gca,'FontName',fontFamily,'FontSize',fontSize)
h = heatmap(xStr, xStr, round(maeNmMatrix,0), ...
    'Colormap', magma, 'CellLabelColor',cellLabelColorStr,'FontSize', fontSize, ...
    'FontName',fontFamily);
h.Position = [0.26 0.36 0.60 0.60];
set(struct(h).NodeChildren(3), 'YTickLabelRotation', 45);
set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45);
xlabel('Test Hertzian Modulus Quartile'); ylabel('Train Hertzian Modulus Quartile');
h.ColorLimits = [min(round(maeNmMatrix,0), [], 'all'), 150];

figure('Name','HERTZIAN MODULUS MAPE', ...
    'Units','inches', 'Position',[1 1 5 5]*1.3);
h = heatmap(xStr,xStr,round(hertzMapeMatrix,0), ...
    'Colormap', magma, 'CellLabelColor',cellLabelColorStr,'FontSize', fontSize, ...
    'FontName',fontFamily);
h.Position = [0.26 0.36 0.60 0.60];
set(struct(h).NodeChildren(3), 'YTickLabelRotation', 45);
set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45);
h.ColorLimits = [min(round(hertzMapeMatrix,0), [], 'all'), 50];
xlabel('Test Hertzian Modulus Quartile'); ylabel('Train Hertzian Modulus Quartile');
%% Save
save('bilstm_does_it_generalize_by_modulus_no_augmentation.mat','K','maeNormMatrix','maeNmMatrix','hertzMapeMatrix', ...
    'mod500MapeMatrix')


%% ------------------------------------------------------------------
%% Local subfunction: doKfoldRegressionInBin
%%  => for bin i-> i we do 5-fold CV on Xbin, Ybin
function metricsCV = doKfoldRegressionInBin(X, Y, goodOrBad, binMask, layers, opts, S)
    % Extract just the bin data
    Xbin = X(:,:,binMask);
    Ybin = Y(binMask);
    Nbin = sum(binMask);

    if Nbin < 2
        metricsCV = struct('maeNorm',NaN,'maeNm',NaN,'hertzMAPE',NaN,'mod500MAPE',NaN);
        return;
    end

    k=5;
    if Nbin< k
        warning('Bin has fewer than %d => skipping K-fold => return NaN.', k);
        metricsCV = struct('maeNorm',NaN,'maeNm',NaN,'hertzMAPE',NaN,'mod500MAPE',NaN);
        return;
    end

    indices = crossvalind('Kfold', Nbin, k);
    maeNormAll   = nan(k,1);
    maeNmAll     = nan(k,1);
    hertzMapeAll = nan(k,1);
    mod500MapeAll= nan(k,1);

    for foldID=1:k
        testMask = (indices==foldID);
        trainMask= ~testMask;
        Xtrain = Xbin(:,:, trainMask);
        Ytrain = Ybin(trainMask);
        Xtest  = Xbin(:,:, testMask);
        Ytest  = Ybin(testMask);
        testIndsLocal = find(testMask);  % local indices in bin

        % Train
        net = dlnetwork(layers);
        Xp  = permute(Xtrain, [1,3,2]);
        dlX = dlarray(Xp, 'CBT');
        nTrainFold = sum(trainMask);
        rp = randperm(nTrainFold);
        dlX = dlX(:,rp,:);
        Ytrain= Ytrain(rp);

        try
            [netFold,info] = trainnet(dlX, Ytrain, net, "mae", opts);
        catch
            warning('Fold %d => error => skip.', foldID);
            continue;
        end

        if isempty(netFold)
            warning('Fold %d => netFold empty => skip.', foldID);
            continue;
        end

        % Evaluate
        globalIndsInBin = find(binMask);   % indices of bin in the original data
        testIndsGlobal  = globalIndsInBin(testIndsLocal);

        outFold = evaluateRegressionSet(netFold, Xtest, Ytest, ...
            S, testIndsGlobal, 500, goodOrBad);
        maeNormAll(foldID)   = outFold.maeNorm;
        maeNmAll(foldID)     = outFold.maeNm;
        hertzMapeAll(foldID) = outFold.hertzMAPE;
        mod500MapeAll(foldID)= outFold.mod500MAPE;
    end

    metricsCV = struct();
    metricsCV.maeNorm   = mean(maeNormAll, 'omitnan');
    metricsCV.maeNm     = mean(maeNmAll, 'omitnan');
    metricsCV.hertzMAPE = mean(hertzMapeAll, 'omitnan');
    metricsCV.mod500MAPE= mean(mod500MapeAll,'omitnan');
end


%% Local subfunction: evaluateRegressionSet
function metricsOut = evaluateRegressionSet(trainedNet, Xtest, Ytest, ...
    S, testIndsGlobal, indentationDepth_nm, goodOrBadOverride)
% EVALUATEREGRESSIONSET
%   Predicts CP => computes MAE(nm), plus Hertz / 500nm MAPE on "good" curves.
%   If S is empty => skip modulus. If goodOrBadOverride is provided => use that
%   instead of S.goodOrBad.
%
% Inputs:
%  - trainedNet: dlnetwork
%  - Xtest, Ytest: test data for this subset
%  - S: struct containing rawExt, rawDefl, minExtValues, maxExtValues, etc.
%  - testIndsGlobal: the absolute indices in the original dataset
%  - indentationDepth_nm: e.g. 500
%  - goodOrBadOverride: optional array for "good" marking
%
% Output:
%  - metricsOut: struct with maeNorm, maeNm, hertzMAPE, mod500MAPE

    metricsOut = struct('maeNorm',NaN,'maeNm',NaN,'hertzMAPE',NaN,'mod500MAPE',NaN);
    if isempty(Xtest) || isempty(Ytest) || isempty(trainedNet)
        return;
    end

    % 1) Predict CP (normalized)
    Xp  = permute(Xtest, [1,3,2]); % [F x N x T]
    dlX = dlarray(Xp,'CBT');
    Ypred= predict(trainedNet, dlX);
    Ypred= extractdata(Ypred)'; % Nx1

    % 2) MAE norm
    errNorm = Ypred - Ytest;
    maeNorm = mean(abs(errNorm),'omitnan');
    metricsOut.maeNorm = maeNorm;

    % 3) MAE in nm (need minExtValues, maxExtValues)
    maeNm = NaN;
    if ~isempty(S) && isfield(S,'minExtValues') && ~isempty(testIndsGlobal)
        mins = S.minExtValues(testIndsGlobal)';
        maxs = S.maxExtValues(testIndsGlobal)';
        predNm= Ypred .* (maxs - mins) + mins;
        trueNm= Ytest .* (maxs - mins) + mins;
        errNm = predNm - trueNm;
        maeNm = mean(abs(errNm),'omitnan');
    end
    metricsOut.maeNm = maeNm;

    % 4) Modulus MAPE => only if S is given
    if isempty(S) || ~isfield(S,'goodOrBad') || ~isfield(S,'rawExt')
        metricsOut.hertzMAPE=NaN; metricsOut.mod500MAPE=NaN;
        return;
    end

    if nargin<7
        goodOrBadLocal = S.goodOrBad(testIndsGlobal);
    else
        goodOrBadLocal = goodOrBadOverride(testIndsGlobal);
    end
    goodMask = (goodOrBadLocal==1);
    if ~any(goodMask)
        % No good => skip
        metricsOut.hertzMAPE=NaN; metricsOut.mod500MAPE=NaN;
        return;
    end

    YPredGood = Ypred(goodMask);
    YTestGood = Ytest(goodMask);
    goodIndsAbs = testIndsGlobal(goodMask);
    % Keep everything.
    goodIndsAbs(:) = 1;

    % Calculate moduli
    if ~all(isfield(S,["rawExt","rawDefl","b","th","R","v","spring_constant"]))
        metricsOut.hertzMAPE=NaN; metricsOut.mod500MAPE=NaN;
        return;
    end

    [Hact, Hpred, M500act, M500pred] = calculateModuli( ...
        S.rawExt, S.rawDefl, YTestGood, YPredGood, goodIndsAbs, ...
        S.minExtValues, S.maxExtValues, S.b, S.th, S.R, S.v, S.spring_constant, ...
        indentationDepth_nm);

    if isempty(Hact)
        metricsOut.hertzMAPE=NaN; metricsOut.mod500MAPE=NaN;
        return;
    end

    maskH = ~isnan(Hact) & ~isnan(Hpred);
    mask5= ~isnan(M500act) & ~isnan(M500pred);

    if any(maskH)
        hzErr= Hpred(maskH) - Hact(maskH);
        hzAPE= abs(hzErr)./abs(Hact(maskH))*100;
        metricsOut.hertzMAPE = mean(hzAPE,'omitnan');
    else
        metricsOut.hertzMAPE=NaN;
    end

    if any(mask5)
        m5Err= M500pred(mask5) - M500act(mask5);
        m5APE= abs(m5Err)./abs(M500act(mask5))*100;
        metricsOut.mod500MAPE = mean(m5APE,'omitnan');
    else
        metricsOut.mod500MAPE=NaN;
    end
end
