function [maeVals, mseVals, mapeVals, bestN, ...
          E_hertz_actual, E_hertz_pred, E_500nm_actual, E_500nm_pred, ...
          badHertzCount, bad500Count, ...
          hertzMSE, hertzMAE, hertzMAPE, ...
          mod500nmMSE, mod500nmMAE, mod500nmMAPE] = evaluate_rov_method_v3()
%EVALUATE_ROV_METHOD
% Evaluates the Ratio of Variance (RoV) method predictions by comparing them to
% the ground-truth normalized contact points (Y).
% Computes metrics (MSE, MAE, MAPE) for each interval size n and outputs results
% for the best-performing n based on MAE (in nm).
%
% Additionally:
%  - Calculates the Hertzian Modulus and the 500 nm Modulus using the external
%    "calculateModuli()" function.
%  - Computes MSE, MAE, and MAPE for both Hertzian and 500 nm moduli.
%  - Produces scatter/histogram plots for the best-performing n and saves them.
%  - Saves all results in a .mat file.

    %% ----------------- Initialization ----------------- %%
    clc;  % Clear command window
    fprintf('Starting evaluation of the RoV Method predictions...\n');
    close all;  % Close all figures

    %% ----------------- Configuration ----------------- %%
    dataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";
    rovPredFile = "predicted_contact_points_rov.mat";

    % Output file for saving the evaluation results
    evaluationOutputFile = 'rov_evaluation_results.mat';


    % Add folder with helper functions
    addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\helperFunctions");

    % Choose whether to save the results
    saveResults = true;

    % Depth at which to extract pointwise modulus (nm)
    indentationDepth_nm = 500;         

    %% ----------------- Load "True" Y and Data ----------------- %%
    if ~isfile(dataFile)
        error('Preprocessed data file not found: %s', dataFile);
    end
    dataStruct = load(dataFile);

    requiredVars = ["Y", "processedExt", "processedDefl", ...
                    "maxExtValues", "minExtValues", "rawExt", "rawDefl", "goodOrBad"];
    for rv = requiredVars
        if ~isfield(dataStruct, rv)
            error('Required variable "%s" not found in %s.', rv, dataFile);
        end
    end

    Y_true = dataStruct.Y;                         
    processedExt = dataStruct.processedExt;
    processedDefl = dataStruct.processedDefl;
    maxExtValues = dataStruct.maxExtValues';  
    minExtValues = dataStruct.minExtValues';  
    rawExt = dataStruct.rawExt;            
    rawDefl = dataStruct.rawDefl;  
    goodOrBad = dataStruct.goodOrBad;

    numCurves = length(Y_true);
    fprintf('Loaded true normalized contact points (Y) with %d samples from "%s".\n', ...
            numCurves, dataFile);

    %% ----------------- Load RoV Predictions ----------------- %%
    if ~isfile(rovPredFile)
        error('Predictions file not found: %s', rovPredFile);
    end
    predStruct = load(rovPredFile);

    if ~isfield(predStruct, 'YPredMatrix') || ~isfield(predStruct, 'nRange')
        error('Variables "YPredMatrix" or "nRange" not found in %s.', rovPredFile);
    end
    YPredMatrix = predStruct.YPredMatrix;   % [numCurves x numN]
    nRange = predStruct.nRange;           

    [numPredCurves, numN] = size(YPredMatrix);
    if numCurves ~= numPredCurves
        error('Mismatch in the number of curves between true and predicted data.');
    end

    fprintf('Loaded RoV predictions with %d samples and %d interval sizes from "%s".\n', ...
            numPredCurves, numN, rovPredFile);

    %% ----------------- Compute CP Error Statistics ----------------- %%
    maeValsNorm   = nan(1, numN);  % MAE (normalized)
    mseValsNorm   = nan(1, numN);  % MSE (normalized)
    mapeValsNorm  = nan(1, numN);  % MAPE (normalized)

    maeVals       = nan(1, numN);  % MAE (nm)
    mseVals       = nan(1, numN);  % MSE (nm)
    mapeVals      = nan(1, numN);  % MAPE (nm)

    for nIdx = 1:numN
        YPred = YPredMatrix(:, nIdx);

        %% --- Normalized errors --- %%
        errorsNorm = YPred - Y_true;
        maeValsNorm(nIdx) = mean(abs(errorsNorm), 'omitnan');
        mseValsNorm(nIdx) = mean(errorsNorm.^2, 'omitnan');

        % MAPE (Normalized), ignoring Y_true == 0
        validNormIdx = (Y_true ~= 0);
        if any(validNormIdx)
            mapeValsNorm(nIdx) = mean( ...
                abs(errorsNorm(validNormIdx) ./ Y_true(validNormIdx)) * 100, 'omitnan');
        end

        %% --- Errors in nm --- %%
        predInNmThis = YPred .* (maxExtValues - minExtValues) + minExtValues;
        trueInNmThis = Y_true .* (maxExtValues - minExtValues) + minExtValues;
        errorsInNm = predInNmThis - trueInNmThis;

        maeVals(nIdx) = mean(abs(errorsInNm), 'omitnan');
        mseVals(nIdx) = mean(errorsInNm.^2, 'omitnan');

        % MAPE (nm), ignoring trueInNm == 0
        validNmIdx = (trueInNmThis ~= 0);
        if any(validNmIdx)
            mapeVals(nIdx) = mean( ...
                abs(errorsInNm(validNmIdx) ./ trueInNmThis(validNmIdx)) * 100, 'omitnan');
        end
    end

    % Find the best n based on MAE (in nm)
    [~, bestNIdx] = min(maeVals);
    bestN = nRange(bestNIdx);

    fprintf('\n=== Best interval size (n): %d ===\n', bestN);
    fprintf('  MSE (normalized): %.6f\n', mseValsNorm(bestNIdx));
    fprintf('  MAE (normalized): %.6f\n', maeValsNorm(bestNIdx));
    fprintf('  MAPE (normalized): %.2f%%\n', mapeValsNorm(bestNIdx));
    fprintf('  MSE (nm): %.6f\n', mseVals(bestNIdx));
    fprintf('  MAE (nm): %.6f\n', maeVals(bestNIdx));
    fprintf('  MAPE (nm): %.2f%%\n\n', mapeVals(bestNIdx));

    %% ----------------- Optionally Save CP Results ----------------- %%
    if saveResults
        fprintf('Saving preliminary CP evaluation results to "%s"...\n', evaluationOutputFile);
        save(evaluationOutputFile, ...
            'maeValsNorm', 'mseValsNorm', 'mapeValsNorm', ...
            'maeVals', 'mseVals', 'mapeVals', ...
            'bestN', 'nRange');
        fprintf('Preliminary CP results saved successfully.\n');
    end

    %% ----------------- Compute Moduli (Hertz & 500 nm) for Best n ----------------- %%
    fprintf('Starting modulus calculations for best n = %d using calculateModuli()...\n', bestN);

    % Extract the best predictions
    YPred_best = YPredMatrix(:, bestNIdx);

    % Evaluate all curves -> dataIdx = true(1, numCurves)
    dataIdx = 1:numCurves;
    % Filter out curves that are low-quality
    %dataIdx = dataIdx(logical(goodOrBad));

    % Call "calculateModuli()" (which must already be on the path or in the same folder)
    [E_hertz_actual, E_hertz_pred, E_500nm_actual, E_500nm_pred] = ...
        calculateModuli( ...
            rawExt, rawDefl, ...
            Y_true, YPred_best, ...
            dataIdx, ...
            minExtValues, maxExtValues, ...
            dataStruct.b, dataStruct.th, dataStruct.R, dataStruct.v, ...
            dataStruct.spring_constant, ...
            indentationDepth_nm);




    % Count how many times the results are NaN
    badHertzCountActual = sum(isnan(E_hertz_actual));
    badHertzCountPred = sum(isnan(E_hertz_pred));
    bad500CountActual = sum(isnan(E_500nm_actual));
    bad500CountPred = sum(isnan(E_500nm_pred));

    fprintf('Hertzian Modulus Calculations Failed (PRED): %d out of %d samples.\n', ...
            badHertzCountPred, numCurves);
    fprintf('500 nm Modulus Calculations Failed (PRED): %d out of %d samples.\n\n', ...
            bad500CountPred, numCurves);
        fprintf('Hertzian Modulus Calculations Failed (ACTUAL): %d out of %d samples.\n', ...
            badHertzCountActual, numCurves);
    fprintf('500 nm Modulus Calculations Failed (ACTUAL:): %d out of %d samples.\n\n', ...
            bad500CountActual, numCurves);


    %% ----------------- Compute MSE, MAE, MAPE for the Moduli ----------------- %%
    % 1) Hertzian Modulus
    hertzErrors = E_hertz_pred - E_hertz_actual;
    hertzMSE = mean(hertzErrors.^2, 'omitnan');
    hertzMAE = mean(abs(hertzErrors), 'omitnan');

    nonzeroHertzIdx = (E_hertz_actual ~= 0 & ~isnan(E_hertz_actual) & ~isnan(E_hertz_pred));
    if any(nonzeroHertzIdx)
        hertzMAPE = mean( ...
            abs(hertzErrors(nonzeroHertzIdx) ./ E_hertz_actual(nonzeroHertzIdx)) * 100, ...
            'omitnan');
    else
        hertzMAPE = NaN;
    end

    % 2) 500 nm Modulus
    mod500Errors = E_500nm_pred - E_500nm_actual;
    mod500nmMSE = mean(mod500Errors.^2, 'omitnan');
    mod500nmMAE = mean(abs(mod500Errors), 'omitnan');

    nonzero500nmIdx = (E_500nm_actual ~= 0 & ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred));
    if any(nonzero500nmIdx)
        mod500nmMAPE = mean( ...
            abs(mod500Errors(nonzero500nmIdx) ./ E_500nm_actual(nonzero500nmIdx)) * 100, ...
            'omitnan');
    else
        mod500nmMAPE = NaN;
    end

    fprintf('=== Hertzian Modulus Statistics ===\n');
    fprintf('MSE:  %.6f kPa^2\n', hertzMSE);
    fprintf('MAE:  %.6f kPa\n', hertzMAE);
    fprintf('MAPE: %.2f%%\n\n', hertzMAPE);

    fprintf('=== 500 nm Modulus Statistics ===\n');
    fprintf('MSE:  %.6f kPa^2\n', mod500nmMSE);
    fprintf('MAE:  %.6f kPa\n', mod500nmMAE);
    fprintf('MAPE: %.2f%%\n\n', mod500nmMAPE);


    %% ----------------- Generate Scatter Plots ----------------- %%
    fprintf('Generating scatter plots for best n = %d...\n', bestN);

    % --- CP: Predicted vs Actual (Normalized) --- %
    figure('Name','RoV Method - Predicted vs Actual (Normalized)','NumberTitle','off');
    scatter(Y_true, YPred_best, 36, 'b', 'filled'); hold on; grid on;
    minVal = min([Y_true(:); YPred_best(:)]);
    maxVal = max([Y_true(:); YPred_best(:)]);
    plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    xlabel('Actual (Normalized)');
    ylabel('Predicted (Normalized)');
    title(sprintf('RoV Method (n = %d)\nMSE=%.4f, MAE=%.4f, MAPE=%.2f%%', ...
                  bestN, mseValsNorm(bestNIdx), maeValsNorm(bestNIdx), mapeValsNorm(bestNIdx)));
    legend('Data points','y = x','Location','best');
    savefig(gcf, sprintf('RoV_Predicted_vs_Actual_Normalized_n%d.fig', bestN));

    % --- CP: Predicted vs Actual (nm) --- %
    trueInNm = Y_true .* (maxExtValues - minExtValues) + minExtValues;
    predInNm = YPred_best .* (maxExtValues - minExtValues) + minExtValues;

    figure('Name','RoV Method - Predicted vs Actual (nm)','NumberTitle','off');
    scatter(trueInNm, predInNm, 36, 'b', 'filled'); hold on; grid on;
    minVal = min([trueInNm(:); predInNm(:)]);
    maxVal = max([trueInNm(:); predInNm(:)]);
    plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    xlabel('Actual (nm)');
    ylabel('Predicted (nm)');
    title(sprintf('RoV Method (n = %d)\nMSE=%.4f nm^2, MAE=%.4f nm, MAPE=%.2f%%', ...
                  bestN, mseVals(bestNIdx), maeVals(bestNIdx), mapeVals(bestNIdx)));
    legend('Data points','y = x','Location','best');
    savefig(gcf, sprintf('RoV_Predicted_vs_Actual_nm_n%d.fig', bestN));

    % --- Hertzian Modulus (kPa): Actual vs Predicted --- %
    figure('Name','RoV Method - Hertzian Modulus: Actual vs Predicted','NumberTitle','off');
    scatter(E_hertz_actual, E_hertz_pred, 36, 'r', 'filled'); hold on; grid on;
    validHertz = ~isnan(E_hertz_actual) & ~isnan(E_hertz_pred);
    if any(validHertz)
        minVal = min([E_hertz_actual(validHertz); E_hertz_pred(validHertz)]);
        maxVal = max([E_hertz_actual(validHertz); E_hertz_pred(validHertz)]);
        plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    end
    xlabel('Hertzian Modulus (Actual) [kPa]');
    ylabel('Hertzian Modulus (Predicted) [kPa]');
    title('Hertzian Modulus: Actual vs Predicted');
    legend('Data points','y = x','Location','best');
    savefig(gcf, sprintf('RoV_Hertzian_Modulus_Actual_vs_Predicted_n%d.fig', bestN));

    % --- 500 nm Modulus (kPa): Actual vs Predicted --- %
    figure('Name','RoV Method - 500 nm Modulus: Actual vs Predicted','NumberTitle','off');
    scatter(E_500nm_actual, E_500nm_pred, 36, 'g', 'filled'); hold on; grid on;
    valid500nm = ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred);
    if any(valid500nm)
        minVal = min([E_500nm_actual(valid500nm); E_500nm_pred(valid500nm)]);
        maxVal = max([E_500nm_actual(valid500nm); E_500nm_pred(valid500nm)]);
        plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    end
    xlabel(sprintf('Modulus at %d nm (Actual) [kPa]', indentationDepth_nm));
    ylabel(sprintf('Modulus at %d nm (Predicted) [kPa]', indentationDepth_nm));
    title(sprintf('Modulus at %d nm: Actual vs Predicted', indentationDepth_nm));
    legend('Data points','y = x','Location','best');
    savefig(gcf, sprintf('RoV_500nm_Modulus_Actual_vs_Predicted_n%d.fig', bestN));

    %% ----------------- Save All Final Results ----------------- %%
    if saveResults
        fprintf('Saving final evaluation (CP + Moduli + MSE/MAE/MAPE) to "%s"...\n', evaluationOutputFile);
        save(evaluationOutputFile, ...
            'E_hertz_actual', 'E_hertz_pred', ...
            'E_500nm_actual', 'E_500nm_pred', ...
            'badHertzCountActual','badHertzCountPred', ...
            'bad500CountActual','bad500CountPred', ...
            'hertzMSE', 'hertzMAE', 'hertzMAPE', ...
            'mod500nmMSE', 'mod500nmMAE', 'mod500nmMAPE', ...
            'trueInNm',"predInNm", ...
            '-append');
        fprintf('All evaluation results saved successfully.\n');
    end

    %% ----------------- Optional: Histograms of Moduli ----------------- %%
    figure('Name','RoV Method - Hertzian Modulus Distribution','NumberTitle','off');
    histogram(E_hertz_actual, 'Normalization', 'pdf', 'FaceColor', 'r', 'EdgeColor', 'k');
    hold on;
    histogram(E_hertz_pred, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'k');
    xlabel('Hertzian Modulus [kPa]');
    ylabel('Probability Density');
    title(sprintf('Hertzian Modulus Distribution (n = %d)', bestN));
    legend('Actual','Predicted','Location','best');
    grid on;
    savefig(gcf, sprintf('RoV_Hertzian_Modulus_Distribution_n%d.fig', bestN));

    figure('Name','RoV Method - 500 nm Modulus Distribution','NumberTitle','off');
    histogram(E_500nm_actual, 'Normalization', 'pdf', 'FaceColor', 'g', 'EdgeColor', 'k');
    hold on;
    histogram(E_500nm_pred, 'Normalization', 'pdf', 'FaceColor', 'c', 'EdgeColor', 'k');
    xlabel(sprintf('Modulus at %d nm [kPa]', indentationDepth_nm));
    ylabel('Probability Density');
    title(sprintf('500 nm Modulus Distribution (n = %d)', bestN));
    legend('Actual','Predicted','Location','best');
    grid on;
    savefig(gcf, sprintf('RoV_500nm_Modulus_Distribution_n%d.fig', bestN));

    fprintf('Scatter plots and modulus distributions generated and saved successfully.\n');
    fprintf('Evaluation of RoV Method completed.\n');
end
