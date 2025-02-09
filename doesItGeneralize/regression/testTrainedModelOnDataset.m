function metricsOut = testTrainedModelOnDataset(trainedNetOrFile, preProcessedDataFileTest, ...
    indentationDepth_nm, saveTestResults, saveFileName)
% TESTTRAINEDMODELONDATASET
%   Loads or uses an already-trained network to predict CPs on a test dataset,
%   and calculates the same CP + modulus metrics as your original code.
%
% Inputs:
%   - trainedNetOrFile:  either a dlnetwork object or a path to a .mat that contains 'trainedNet'
%   - preProcessedDataFileTest: .mat file with X, Y, rawExt, rawDefl, goodOrBad, etc.
%   - indentationDepth_nm: e.g. 500 nm for the pointwise modulus
%   - saveTestResults: bool, if true => saves the results
%   - saveFileName: path to .mat for saving
%
% Output (metricsOut) is a struct with fields:
%   metricsOut.mseTestNorm, metricsOut.maeTestNorm, ...
%   metricsOut.mseTestNm,   metricsOut.maeTestNm,   ...
%   metricsOut.hertzMAE,   metricsOut.hertzMAPE,   etc...
%
% Example usage:
%   m = testTrainedModelOnDataset('trainedModel_Tubules.mat', ...
%           'processed_features_for_regression_HEPG4.mat', 500, true, 'testResults.mat');

    %% 1) Load or accept the trained network
    if ischar(trainedNetOrFile) || isstring(trainedNetOrFile)
        fprintf('Loading trained model from "%s"...\n', trainedNetOrFile);
        tmp = load(trainedNetOrFile, 'trainedNet');
        if ~isfield(tmp, 'trainedNet')
            error('File "%s" does not contain "trainedNet".', trainedNetOrFile);
        end
        trainedNet = tmp.trainedNet;
    else
        % Assume it's already a dlnetwork (or DAGNetwork) object
        trainedNet = trainedNetOrFile;
    end

    %% 2) Load test data
    fprintf('Loading test data from "%s"...\n', preProcessedDataFileTest);
    testData = load(preProcessedDataFileTest);

    requiredFieldsPD = {'X','Y','maxExtValues','minExtValues','rawExt','rawDefl', ...
                        'fileIndices','fileRow','fileCol','b','th','R','v','spring_constant','goodOrBad'};
    for iField = 1:length(requiredFieldsPD)
        if ~isfield(testData, requiredFieldsPD{iField})
            error('Field "%s" missing from test data "%s".', requiredFieldsPD{iField}, preProcessedDataFileTest);
        end
    end

    XTestFull = testData.X;
    YTestFull = testData.Y;
    numSamplesTest = size(XTestFull, 3);
    testIndices = 1:numSamplesTest;  % just 1..N for reference

    %% 3) Predict with the trained network
    fprintf('Predicting on %d test samples...\n', numSamplesTest);
    XTestPermuted = permute(XTestFull, [1, 3, 2]); % [C x B x T]
    XTestDL = dlarray(XTestPermuted, 'CBT');
    YPredTest = predict(trainedNet, XTestDL);
    YPredTest = extractdata(YPredTest)';  % Nx1

    %% 4) Calculate CP metrics (Normalized + nm)
    YTest = YTestFull;
    testErrorsNorm = YPredTest - YTest;
    mseTestNorm = mean(testErrorsNorm.^2);
    maeTestNorm = mean(abs(testErrorsNorm));

    maxExtValuesTest = testData.maxExtValues(testIndices)';
    minExtValuesTest = testData.minExtValues(testIndices)';

    % Convert normalized CPs to nm
    predTestNm = YPredTest .* (maxExtValuesTest - minExtValuesTest) + minExtValuesTest;
    trueTestNm = YTest .* (maxExtValuesTest - minExtValuesTest) + minExtValuesTest;

    testErrorsNm = predTestNm - trueTestNm;
    mseTestNm = mean(testErrorsNm.^2);
    maeTestNm = mean(abs(testErrorsNm));

    %% 5) Modulus metrics on "good" curves only
    goodOrBadTest = testData.goodOrBad;
    goodIndicesTest = testIndices(goodOrBadTest == 1);

    if isempty(goodIndicesTest)
        warning('No good curves found in test set, skipping modulus calculations...');
        HertzianModulusActualTest    = [];
        HertzianModulusPredictedTest = [];
        Modulus500nmActualTest       = [];
        Modulus500nmPredictedTest    = [];
        mod500MSE  = NaN; mod500MAE  = NaN; mod500MAPE  = NaN;
        hertzMSE   = NaN; hertzMAE   = NaN; hertzMAPE   = NaN;
    else
        [HertzianModulusActualTest, HertzianModulusPredictedTest, ...
         Modulus500nmActualTest, Modulus500nmPredictedTest] = calculateModuli( ...
            testData.rawExt, testData.rawDefl, ...
            YTest(goodOrBadTest==1), ...
            YPredTest(goodOrBadTest==1), ...
            goodIndicesTest, ...
            testData.minExtValues, testData.maxExtValues, ...
            testData.b, testData.th, testData.R, testData.v, testData.spring_constant, ...
            indentationDepth_nm);

        % 500 nm modulus
        mod500Errors = Modulus500nmPredictedTest - Modulus500nmActualTest;
        mod500SE = mod500Errors.^2;
        mod500AE = abs(mod500Errors);
        mod500APE = abs(mod500Errors ./ Modulus500nmActualTest) * 100;

        mod500MSE  = mean(mod500SE, 'omitnan');
        mod500MAE  = mean(mod500AE, 'omitnan');
        mod500MAPE = mean(mod500APE, 'omitnan');

        % Hertz modulus
        hertzErrors = HertzianModulusPredictedTest - HertzianModulusActualTest;
        hertzSE = hertzErrors.^2;
        hertzAE = abs(hertzErrors);
        hertzAPE = abs(hertzErrors ./ HertzianModulusActualTest) * 100;

        hertzMSE  = mean(hertzSE, 'omitnan');
        hertzMAE  = mean(hertzAE, 'omitnan');
        hertzMAPE = mean(hertzAPE, 'omitnan');
    end

    %% 6) Gather into a struct
    metricsOut = struct();
    metricsOut.mseTestNorm = mseTestNorm;
    metricsOut.maeTestNorm = maeTestNorm;
    metricsOut.mseTestNm   = mseTestNm;
    metricsOut.maeTestNm   = maeTestNm;

    metricsOut.predTestNm  = predTestNm;
    metricsOut.trueTestNm  = trueTestNm;

    metricsOut.HertzianModulusActualTest    = HertzianModulusActualTest;
    metricsOut.HertzianModulusPredictedTest = HertzianModulusPredictedTest;
    metricsOut.Modulus500nmActualTest       = Modulus500nmActualTest;
    metricsOut.Modulus500nmPredictedTest    = Modulus500nmPredictedTest;

    metricsOut.mod500MSE  = mod500MSE;
    metricsOut.mod500MAE  = mod500MAE;
    metricsOut.mod500MAPE = mod500MAPE;

    metricsOut.hertzMSE   = hertzMSE;
    metricsOut.hertzMAE   = hertzMAE;
    metricsOut.hertzMAPE  = hertzMAPE;

    %% 7) (Optional) Save results
    if saveTestResults
        if nargin < 5 || isempty(saveFileName)
            saveFileName = sprintf('testResults_%s.mat', datestr(now,'yyyymmdd_HHMM'));
        end
        fprintf('Saving test metrics to "%s"...\n', saveFileName);
        save(saveFileName, '-struct', 'metricsOut');
    end

    fprintf('Test done. MSEnorm=%.5f, MAEnm=%.3f nm, MAPE(500nm)=%.2f%%, MAPE(Hertz)=%.2f%%\n', ...
        mseTestNorm, maeTestNm, mod500MAPE, hertzMAPE);
end
