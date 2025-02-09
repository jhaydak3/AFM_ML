function [mseVal, maeVal, mseValNm, maeValNm, ...
          E_hertz_actual, E_hertz_pred, ...
          E_500nm_actual, E_500nm_pred, ...
          badHertzCount, bad500Count, ...
          hertzMSE, hertzMAE, hertzMAPE, ...
          mod500nmMSE, mod500nmMAE, mod500nmMAPE] ...
          = evaluate_model_piecewise_custom_regression_v4()
%EVALUATE_MODEL_PIECEWISE_CUSTOM_REGRESSION_V2
%
% Evaluates the accuracy of piecewise model predictions against ground-truth
% normalized contact points (Y) from preprocessed AFM data. Additionally, it
% computes the Hertzian Modulus and the Modulus at a specified indentation depth
% (defaulted to 500 nm) for both actual and predicted contact points, using
% the calculateModuli() function.
%
% The function performs the following:
%   1. Loads true and predicted normalized contact points.
%   2. Computes Mean Squared Error (MSE) and Mean Absolute Error (MAE) in both
%      normalized units and physical units (nm) for the contact points.
%   3. Calls calculateModuli() to compute Hertzian Modulus and Modulus at 500 nm.
%   4. Computes MSE, MAE, and MAPE for both moduli.
%   5. Saves all relevant results (including modulus metrics) to a .mat file.
%   6. Generates scatter plots comparing actual vs predicted results.
%
% Outputs:
%   - mseVal:          MSE of contact points (normalized)
%   - maeVal:          MAE of contact points (normalized)
%   - mseValNm:        MSE of contact points (nm)
%   - maeValNm:        MAE of contact points (nm)
%   - E_hertz_actual:  Hertzian Modulus (Actual) [kPa]
%   - E_hertz_pred:    Hertzian Modulus (Predicted) [kPa]
%   - E_500nm_actual:  Modulus at 500 nm (Actual) [kPa]
%   - E_500nm_pred:    Modulus at 500 nm (Predicted) [kPa]
%   - badHertzCount:   Number of failed Hertzian modulus calculations
%   - bad500Count:     Number of failed 500 nm modulus calculations
%   - hertzMSE:        MSE of Hertzian modulus [kPa^2]
%   - hertzMAE:        MAE of Hertzian modulus [kPa]
%   - hertzMAPE:       MAPE of Hertzian modulus [%]
%   - mod500nmMSE:     MSE of 500 nm modulus [kPa^2]
%   - mod500nmMAE:     MAE of 500 nm modulus [kPa]
%   - mod500nmMAPE:    MAPE of 500 nm modulus [%]
%
% Author: Your Name
% Date: January 6, 2025

    %% ----------------- Initialization ----------------- %%
    clc;  % Clear command window
    fprintf('Starting evaluation of the Piecewise Model predictions...\n');

    %% ----------------- Configuration ----------------- %%
    % Paths to data and prediction files
    dataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";
    piecewisePredFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\AzelogluCosta2011\predicted_contact_points_piecewise_fit_deflectionFraction.mat"

    % Add folder with helper functions
    addpath("C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\helperFunctions");


    % Output file for saving evaluation results
    evaluationOutputFile = "piecewise_evaluation_results.mat";

    % Choose whether to save the results
    saveResults = true;

    % Depth at which to extract pointwise modulus (nm)
    indentationDepth_nm = 500;

    %% ----------------- Load "True" Data ----------------- %%
    if ~isfile(dataFile)
        error('Data file not found: %s', dataFile);
    end
    dataStruct = load(dataFile);

    % Check for required variables
    requiredVars = ["Y", "maxExtValues", "minExtValues", ...
                    "rawExt", "rawDefl", "R", "b", "th", ...
                    "spring_constant", "v", "goodOrBad"];
    for rv = requiredVars
        if ~isfield(dataStruct, rv)
            error('Missing required variable "%s" in dataFile.', rv);
        end
    end

    % Extract data
    Y_true          = dataStruct.Y;              % [numCurves x 1] normalized CP
    maxExtValues    = dataStruct.maxExtValues';  % [1 x numCurves]
    minExtValues    = dataStruct.minExtValues';  % [1 x numCurves]
    rawExt          = dataStruct.rawExt;         % {1 x numCurves} cell array
    rawDefl         = dataStruct.rawDefl;        % {1 x numCurves} cell array
    R_all           = dataStruct.R;              % [1 x numCurves], tip radius (nm)
    b_all           = dataStruct.b;              % [1 x numCurves], blunt radius (nm)
    th_all          = dataStruct.th;             % [1 x numCurves], half-opening angle (rad)
    k_all           = dataStruct.spring_constant;% [1 x numCurves] (N/m)
    v_all           = dataStruct.v;              % [1 x numCurves] Poisson ratio
    goodOrBad       = dataStruct.goodOrBad;      % [1 x numCurves], quality of curve


    numSamples_true = length(Y_true);
    fprintf('Loaded true contact points with %d samples.\n', numSamples_true);

    %% ----------------- Load Predictions ----------------- %%
    if ~isfile(piecewisePredFile)
        error('Predictions file not found: %s', piecewisePredFile);
    end
    predStruct = load(piecewisePredFile);

    % Check for predicted variable
    if ~isfield(predStruct, 'YPred_new')
        error('Variable "YPred_new" not found in %s.', piecewisePredFile);
    end
    YPred_piecewise = predStruct.YPred_new;  % [numCurves x 1] normalized
    numSamples_pred = length(YPred_piecewise);

    fprintf('Loaded piecewise predictions with %d samples.\n', numSamples_pred);

    % Check consistency
    if numSamples_true ~= numSamples_pred
        error('Mismatch in number of curves: %d (true) vs %d (pred).', ...
              numSamples_true, numSamples_pred);
    end

    %% ----------------- Compute Basic Contact-Point Errors ----------------- %%
    errorsNorm = YPred_piecewise - Y_true;
    mseVal = mean(errorsNorm.^2, 'omitnan');
    maeVal = mean(abs(errorsNorm), 'omitnan');

    % Convert normalized CP to nm
    predInNm = YPred_piecewise .* (maxExtValues - minExtValues) + minExtValues;
    trueInNm = Y_true .* (maxExtValues - minExtValues) + minExtValues;
    errorsInNm = predInNm - trueInNm;

    mseValNm = mean(errorsInNm.^2, 'omitnan');
    maeValNm = mean(abs(errorsInNm), 'omitnan');

    fprintf('\n=== Piecewise Model Evaluation (Contact Points) ===\n');
    fprintf('  MSE (normalized): %.6f\n', mseVal);
    fprintf('  MAE (normalized): %.6f\n', maeVal);
    fprintf('  MSE (nm):        %.6f\n', mseValNm);
    fprintf('  MAE (nm):        %.6f\n', maeValNm);

    %% ----------------- Compute Moduli via calculateModuli() ----------------- %%
    % Create a logical index that includes all curves
    dataIdx = 1:length(rawExt);
    % Filter out low quality curves
    %dataIdx = dataIdx(logical(goodOrBad));

    fprintf('\nCalling calculateModuli() for Hertzian and 500 nm modulus...\n');
    [E_hertz_actual, E_hertz_pred, E_500nm_actual, E_500nm_pred] = ...
        calculateModuli( ...
            rawExt, rawDefl, ...
            Y_true, YPred_piecewise, ...
            dataIdx, ...
            minExtValues, maxExtValues, ...
            b_all, th_all, R_all, v_all, ...
            k_all, ...
            indentationDepth_nm);

    % Count how many times the results are NaN
    badHertzCountActual = sum(isnan(E_hertz_actual));
    badHertzCountPred = sum(isnan(E_hertz_pred));
    bad500CountActual = sum(isnan(E_500nm_actual));
    bad500CountPred = sum(isnan(E_500nm_pred));

    fprintf('Hertzian Modulus Calculations Failed (PRED): %d out of %d samples.\n', ...
            badHertzCountPred, length(dataIdx));
    fprintf('500 nm Modulus Calculations Failed (PRED): %d out of %d samples.\n\n', ...
            bad500CountPred, length(dataIdx));
        fprintf('Hertzian Modulus Calculations Failed (ACTUAL): %d out of %d samples.\n', ...
            badHertzCountActual, length(dataIdx));
    fprintf('500 nm Modulus Calculations Failed (ACTUAL:): %d out of %d samples.\n\n', ...
            bad500CountActual, length(dataIdx));

    %% ----------------- Compute Error Metrics (Hertz & 500 nm) ----------------- %%
    % Hertzian Modulus Errors
    hertzErrors = E_hertz_pred - E_hertz_actual;
    hertzMSE = mean(hertzErrors.^2, 'omitnan');
    hertzMAE = mean(abs(hertzErrors), 'omitnan');

    nonzeroHertzIdx = (E_hertz_actual ~= 0 & ~isnan(E_hertz_actual) & ~isnan(E_hertz_pred));
    if any(nonzeroHertzIdx)
        hertzMAPE = mean(abs(hertzErrors(nonzeroHertzIdx) ./ E_hertz_actual(nonzeroHertzIdx))) * 100;
    else
        hertzMAPE = NaN;
    end

    % 500 nm Modulus Errors
    mod500Errors = E_500nm_pred - E_500nm_actual;
    mod500nmMSE  = mean(mod500Errors.^2, 'omitnan');
    mod500nmMAE  = mean(abs(mod500Errors), 'omitnan');

    nonzero500Idx = (E_500nm_actual ~= 0 & ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred));
    if any(nonzero500Idx)
        mod500nmMAPE = mean(abs(mod500Errors(nonzero500Idx) ./ E_500nm_actual(nonzero500Idx))) * 100;
    else
        mod500nmMAPE = NaN;
    end

    fprintf('=== Hertzian Modulus Statistics ===\n');
    fprintf('  MSE:  %.6f kPa^2\n', hertzMSE);
    fprintf('  MAE:  %.6f kPa\n', hertzMAE);
    fprintf('  MAPE: %.2f%%\n\n', hertzMAPE);

    fprintf('=== 500 nm Modulus Statistics ===\n');
    fprintf('  MSE:  %.6f kPa^2\n', mod500nmMSE);
    fprintf('  MAE:  %.6f kPa\n', mod500nmMAE);
    fprintf('  MAPE: %.2f%%\n\n', mod500nmMAPE);

    %% ----------------- (Optional) Save All Results ----------------- %%
    if saveResults
        fprintf('Saving evaluation results to "%s"...\n', evaluationOutputFile);
        save(evaluationOutputFile, ...
             'mseVal','maeVal','mseValNm','maeValNm', ...
             'Y_true','YPred_piecewise','predInNm','trueInNm','errorsInNm', ...
             'E_hertz_actual','E_hertz_pred','E_500nm_actual','E_500nm_pred', ...
             'badHertzCountActual',"badHertzCountPred","bad500CountPred",'bad500CountActual', ...
             'hertzMSE','hertzMAE','hertzMAPE', ...
             'mod500nmMSE','mod500nmMAE','mod500nmMAPE', ...
             '-v7.3');
        fprintf('Evaluation results saved successfully.\n');
    end

    %% ----------------- Generate Scatter Plots ----------------- %%
    fprintf('Generating scatter plots...\n');

    % 1) Predicted vs Actual (Normalized)
    figure('Name','Piecewise Model - Predicted vs Actual (Normalized)','NumberTitle','off');
    scatter(Y_true, YPred_piecewise, 36, 'b', 'filled'); hold on; grid on;
    minVal = min([Y_true(:); YPred_piecewise(:)]);
    maxVal = max([Y_true(:); YPred_piecewise(:)]);
    plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    xlabel('Actual (Normalized)');
    ylabel('Predicted (Normalized)');
    title(sprintf('Piecewise Model (MSE=%.4f, MAE=%.4f)', mseVal, maeVal));
    legend('Data points','y = x','Location','best');
    savefig(gcf, 'PiecewiseModel_Predicted_vs_Actual_Normalized.fig');

    % 2) Predicted vs Actual (nm)
    figure('Name','Piecewise Model - Predicted vs Actual (nm)','NumberTitle','off');
    scatter(trueInNm, predInNm, 36, 'b', 'filled'); hold on; grid on;
    minVal = min([trueInNm(:); predInNm(:)]);
    maxVal = max([trueInNm(:); predInNm(:)]);
    plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    xlabel('Actual (nm)');
    ylabel('Predicted (nm)');
    title(sprintf('Piecewise Model (MSE=%.4f nm^2, MAE=%.4f nm)', mseValNm, maeValNm));
    legend('Data points','y = x','Location','best');
    savefig(gcf, 'PiecewiseModel_Predicted_vs_Actual_nm.fig');

    % 3) Hertzian Modulus (Actual vs Predicted)
    figure('Name','Hertzian Modulus - Actual vs Predicted','NumberTitle','off');
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
    savefig(gcf, 'Hertzian_Modulus_Actual_vs_Predicted.fig');

    % 4) 500 nm Modulus (Actual vs Predicted)
    figure('Name','Modulus at 500 nm - Actual vs Predicted','NumberTitle','off');
    scatter(E_500nm_actual, E_500nm_pred, 36, 'g', 'filled'); hold on; grid on;
    valid500 = ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred);
    if any(valid500)
        minVal = min([E_500nm_actual(valid500); E_500nm_pred(valid500)]);
        maxVal = max([E_500nm_actual(valid500); E_500nm_pred(valid500)]);
        plot([minVal maxVal],[minVal maxVal],'k--','LineWidth',1.5);
    end
    xlabel(sprintf('Modulus at %d nm (Actual) [kPa]', indentationDepth_nm));
    ylabel(sprintf('Modulus at %d nm (Predicted) [kPa]', indentationDepth_nm));
    title(sprintf('Modulus at %d nm: Actual vs Predicted', indentationDepth_nm));
    legend('Data points','y = x','Location','best');
    savefig(gcf, sprintf('Modulus_at_%dnm_Actual_vs_Predicted.fig', indentationDepth_nm));

    fprintf('Scatter plots generated and saved successfully.\n');
    fprintf('Evaluation completed.\n');
end
