function metricsOut = testTrainedModelOnDataset_sub(trainedNet, Xtest, Ytest, dataStruct, testInds, indentationDepth_nm)
% TESTTRAINEDMODELONDATASET_SUB
%   A "real" metric calculation function, replacing dummy placeholders.
%
% Inputs:
%   - trainedNet : your dlnetwork
%   - Xtest, Ytest: a subset of the data
%   - dataStruct : a struct that has rawExt, rawDefl, goodOrBad, etc.
%   - testInds   : the absolute indices in the original dataset
%   - indentationDepth_nm: for modulus calculations
%
% Output:
%   metricsOut   : struct with fields:
%       .maeTestNorm
%       .maeTestNm
%       .mseTestNorm
%       .mseTestNm
%       .hertzMAPE
%       .mod500MAPE
%       .hertzMSE, .hertzMAE, etc.
%       (add any others you want)

testOnGoodCurvesOnly = false;

metricsOut = struct();

% Check for empties
if isempty(trainedNet) || isempty(Xtest) || isempty(Ytest)
    metricsOut.maeTestNorm = NaN;
    metricsOut.mseTestNorm = NaN;
    metricsOut.maeTestNm   = NaN;
    metricsOut.mseTestNm   = NaN;
    metricsOut.hertzMAPE   = NaN;
    metricsOut.mod500MAPE  = NaN;
    metricsOut.hertzMAE    = NaN;
    metricsOut.hertzMSE    = NaN;
    metricsOut.mod500MAE   = NaN;
    metricsOut.mod500MSE   = NaN;
    return;
end

% 1) Predict
%  - Xtest shape = [features x seqLength x #samples]
% Convert to dlarray
XTestPerm = permute(Xtest,[1,3,2]);  % [C x B x T]
dlXTest   = dlarray(XTestPerm, 'CBT');
YPred = predict(trainedNet, dlXTest);
YPred = extractdata(YPred)';  % shape => [#samples x 1]

% 2) Compute CP metrics (normalized)
testErrorsNorm = YPred - Ytest;
mseTestNorm = mean(testErrorsNorm.^2);
maeTestNorm = mean(abs(testErrorsNorm));

metricsOut.mseTestNorm = mseTestNorm;
metricsOut.maeTestNorm = maeTestNorm;

% 3) Convert to nm
%  - We assume dataStruct.minExtValues, dataStruct.maxExtValues are 1xN
%    or Nx1. We extract the relevant subset for testInds
minVals = dataStruct.minExtValues(testInds)';
maxVals = dataStruct.maxExtValues(testInds)';

predTestNm = YPred .* (maxVals - minVals) + minVals;
trueTestNm = Ytest .* (maxVals - minVals) + minVals;

testErrorsNm = predTestNm - trueTestNm;
mseTestNm = mean(testErrorsNm.^2);
maeTestNm = mean(abs(testErrorsNm));
metricsOut.mseTestNm = mseTestNm;
metricsOut.maeTestNm = maeTestNm;

fprintf('CP metrics: MSE(norm)=%.3f, MAE(norm)=%.3f, MSE(nm)=%.3f, MAE(nm)=%.3f\n',...
    mseTestNorm, maeTestNorm, mseTestNm, maeTestNm);

% 4) Filter for good curves => only compute modulus for those
if testOnGoodCurvesOnly
    goodMask = dataStruct.goodOrBad(testInds) == 1;
    goodIndsAbs = testInds(goodMask);  % absolute indices
    if ~any(goodMask)
        % no good curves => can't compute modulus
        metricsOut.hertzMAPE  = NaN;
        metricsOut.mod500MAPE = NaN;
        metricsOut.hertzMAE   = NaN;
        metricsOut.hertzMSE   = NaN;
        metricsOut.mod500MAE  = NaN;
        metricsOut.mod500MSE  = NaN;
        return;
    end

    YPredGood = YPred(goodMask);
    YTestGood = Ytest(goodMask);

    % 5) compute Moduli
    [HertzAct, HertzPred, Mod500Act, Mod500Pred] = calculateModuli( ...
        dataStruct.rawExt, dataStruct.rawDefl, ...
        YTestGood, YPredGood, ...
        goodIndsAbs, ...
        dataStruct.minExtValues, dataStruct.maxExtValues, ...
        dataStruct.b, dataStruct.th, dataStruct.R, dataStruct.v, dataStruct.spring_constant, ...
        indentationDepth_nm);
else
    [HertzAct, HertzPred, Mod500Act, Mod500Pred] = calculateModuli( ...
        dataStruct.rawExt, dataStruct.rawDefl, ...
        Ytest, YPred, ...
        testInds, ...
        dataStruct.minExtValues, dataStruct.maxExtValues, ...
        dataStruct.b, dataStruct.th, dataStruct.R, dataStruct.v, dataStruct.spring_constant, ...
        indentationDepth_nm);
end

% remove NaNs from actual => can't compute MAPE on those
maskValidHertz = ~isnan(HertzAct) & ~isnan(HertzPred);
maskValid500   = ~isnan(Mod500Act) & ~isnan(Mod500Pred);

% 6) MSE, MAE, MAPE for Hertz
hertzErrors = HertzPred(maskValidHertz) - HertzAct(maskValidHertz);
hertzMSE = mean(hertzErrors.^2);
hertzMAE = mean(abs(hertzErrors));
hertzAPE = abs(hertzErrors) ./ abs(HertzAct(maskValidHertz)) * 100;
hertzMAPE= mean(hertzAPE);

metricsOut.hertzMSE  = hertzMSE;
metricsOut.hertzMAE  = hertzMAE;
metricsOut.hertzMAPE = hertzMAPE;

% 7) MSE, MAE, MAPE for 500 nm
mod500Errors = Mod500Pred(maskValid500) - Mod500Act(maskValid500);
mod500MSE = mean(mod500Errors.^2);
mod500MAE = mean(abs(mod500Errors));
mod500APE = abs(mod500Errors)./abs(Mod500Act(maskValid500))*100;
mod500MAPE= mean(mod500APE);

metricsOut.mod500MSE  = mod500MSE;
metricsOut.mod500MAE  = mod500MAE;
metricsOut.mod500MAPE = mod500MAPE;

if testOnGoodCurvesOnly
    fprintf('Modulus metrics (good only): Hertz MSE=%.3f, MAE=%.3f, MAPE=%.2f%%\n', ...
        hertzMSE, hertzMAE, hertzMAPE);
    fprintf('                          500nm MSE=%.3f, MAE=%.3f, MAPE=%.2f%%\n', ...
        mod500MSE, mod500MAE, mod500MAPE);
else
    fprintf('Modulus metrics: Hertz MSE=%.3f, MAE=%.3f, MAPE=%.2f%%\n', ...
        hertzMSE, hertzMAE, hertzMAPE);
    fprintf('                          500nm MSE=%.3f, MAE=%.3f, MAPE=%.2f%%\n', ...
        mod500MSE, mod500MAE, mod500MAPE);
end

end
