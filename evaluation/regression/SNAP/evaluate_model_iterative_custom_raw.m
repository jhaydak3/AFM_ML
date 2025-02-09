function [mseVal, maeVal, mseValNm, maeValNm, ...
          E_hertz_actual, E_hertz_pred, ...
          E_500nm_actual, E_500nm_pred, ...
          badHertzCount, bad500Count, ...
          hertzMSE, hertzMAE, hertzMAPE, ...
          mod500nmMSE, mod500nmMAE, mod500nmMAPE] = ...
         evaluate_model_iterative_custom_raw()
% EVALUATE_MODEL_ITERATIVE_CUSTOM_RAW
%
% Evaluates the "iterative CP" predictions (from predict_regression_iterative_custom_raw.m)
% against the ground truth contact points and calculates error metrics:
%   - MSE, MAE in normalized CP space
%   - MSE, MAE in nm
%   - Hertz and 500 nm modulus errors (if desired)
%
% It loads:
%   1) The main .mat file with true Y, raw data, etc.
%   2) The "predicted_cp_iterative_raw.mat" file with YPred_new, z0_nm, E_Pa.
%
% Then computes:
%   - MSE, MAE (fractional)
%   - MSE, MAE (nm)
%   - Hertz & 500 nm modulus (optional) -> MSE, MAE, MAPE
%


    %% ----------------- Configuration ----------------- %%
    dataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";
    iterPredFile = "predicted_cp_iterative_raw.mat";  % from predict_regression_iterative_custom_raw.m
    
    % Where to save the final evaluation metrics
    evaluationOutputFile = "iterative_evaluation_results.mat";
    
    % Indentation depth for the "500 nm" modulus (if you want that)
    indentationDepth_nm = 500;
    
    % Whether to save the results
    saveResults = true;

    %% ----------------- Load True Data ----------------- %%
    if ~isfile(dataFile)
        error('Data file not found: %s', dataFile);
    end
    D = load(dataFile);
    
    % Required fields
    reqVars = {'Y','minExtValues','maxExtValues','rawExt','rawDefl',...
               'R','b','th','spring_constant','v','goodOrBad'};
    for r = 1:numel(reqVars)
        if ~isfield(D, reqVars{r})
            error('Missing field "%s" in %s', reqVars{r}, dataFile);
        end
    end
    
    Y_true          = D.Y;                 % normalized CP (Nx1)
    minExtValues    = D.minExtValues;      % 1xN
    maxExtValues    = D.maxExtValues;      % 1xN
    rawExt          = D.rawExt;            % {1 x N}
    rawDefl         = D.rawDefl;           % {1 x N}
    R_all           = D.R;                 % tip radius
    b_all           = D.b;                 % blunt radius
    th_all          = D.th;                % half-angle (rad)
    k_all           = D.spring_constant;   % N/nm
    v_all           = D.v;                 % Poisson ratio
    goodOrBad       = D.goodOrBad;         % [1 x N]

    nCurves = numel(Y_true);
    fprintf('Loaded ground-truth data for %d curves.\n', nCurves);

    %% ----------------- Load Predicted Data ----------------- %%
    if ~isfile(iterPredFile)
        error('Prediction file not found: %s', iterPredFile);
    end
    P = load(iterPredFile);  % should have YPred_new, z0_nm, E_Pa
    if ~isfield(P,'YPred_new') || ~isfield(P,'z0_nm') || ~isfield(P,'E_Pa')
        error('Prediction file must contain YPred_new, z0_nm, E_Pa.');
    end
    
    YPred_new = P.YPred_new;   % Nx1
    z0_nm     = P.z0_nm;       % Nx1
    E_Pa      = P.E_Pa;        % Nx1
    
    if numel(YPred_new) ~= nCurves
        error('Mismatch in # of curves: YPred_new has %d, data has %d', ...
              numel(YPred_new), nCurves);
    end
    fprintf('Loaded iterative predictions for %d curves.\n', numel(YPred_new));

    %% ----------------- Contact-Point Errors ----------------- %%
    % 1) Normalized error
    errorsNorm = YPred_new - Y_true;
    mseVal = mean(errorsNorm.^2,'omitnan');
    maeVal = mean(abs(errorsNorm),'omitnan');
    
    % 2) Convert predicted fraction to nm
    predInNm = YPred_new .* (maxExtValues' - minExtValues') + minExtValues';
    trueInNm = Y_true     .* (maxExtValues' - minExtValues') + minExtValues';
    errorsNm = predInNm - trueInNm;
    mseValNm = mean(errorsNm.^2,'omitnan');
    maeValNm = mean(abs(errorsNm),'omitnan');
    
    fprintf('\n=== Iterative Model Contact-Point Evaluation ===\n');
    fprintf('MSE (norm): %.6f\n', mseVal);
    fprintf('MAE (norm): %.6f\n', maeVal);
    fprintf('MSE (nm):   %.6f\n', mseValNm);
    fprintf('MAE (nm):   %.6f\n', maeValNm);

    %% ----------------- Optional: Compute Modulus Errors ----------------- %%
    fprintf('\nComputing Hertz & 500 nm moduli... (using calculateModuli)\n');

    % Build an index of all curves (or filter out "bad" ones if you like)
    dataIdx = 1:nCurves;

    [E_hertz_actual, E_hertz_pred, E_500nm_actual, E_500nm_pred] = ...
        calculateModuli( ...
            rawExt, rawDefl, ...
            Y_true, YPred_new, ...
            dataIdx, ...
            minExtValues, maxExtValues, ...
            b_all, th_all, R_all, v_all, ...
            k_all, ...
            indentationDepth_nm);

    % Count how many are NaN
    badHertzCount = sum(isnan(E_hertz_pred) | isnan(E_hertz_actual));
    bad500Count   = sum(isnan(E_500nm_pred) | isnan(E_500nm_actual));

    fprintf('Hertz: %d/%d curves had NaN results.\n', badHertzCount, nCurves);
    fprintf('500nm: %d/%d curves had NaN results.\n', bad500Count, nCurves);

    % Hertz errors
    hertzErrors = E_hertz_pred - E_hertz_actual;
    hertzMSE = mean(hertzErrors.^2, 'omitnan');
    hertzMAE = mean(abs(hertzErrors), 'omitnan');
    idxNonzero = (E_hertz_actual ~= 0 & ~isnan(E_hertz_actual) & ~isnan(E_hertz_pred));
    if any(idxNonzero)
        hertzMAPE = mean(abs(hertzErrors(idxNonzero)./E_hertz_actual(idxNonzero))) * 100;
    else
        hertzMAPE = NaN;
    end

    % 500 nm errors
    mod500Errors = E_500nm_pred - E_500nm_actual;
    mod500nmMSE = mean(mod500Errors.^2, 'omitnan');
    mod500nmMAE = mean(abs(mod500Errors), 'omitnan');
    idxNonzero2 = (E_500nm_actual ~= 0 & ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred));
    if any(idxNonzero2)
        mod500nmMAPE = mean(abs(mod500Errors(idxNonzero2)./E_500nm_actual(idxNonzero2)))*100;
    else
        mod500nmMAPE = NaN;
    end

    fprintf('\n=== Hertzian Modulus Errors (kPa) ===\n');
    fprintf('MSE:  %.6f\n', hertzMSE);
    fprintf('MAE:  %.6f\n', hertzMAE);
    fprintf('MAPE: %.6f%%\n', hertzMAPE);

    fprintf('\n=== 500 nm Modulus Errors (kPa) ===\n');
    fprintf('MSE:  %.6f\n', mod500nmMSE);
    fprintf('MAE:  %.6f\n', mod500nmMAE);
    fprintf('MAPE: %.6f%%\n', mod500nmMAPE);

    %% ----------------- (Optional) Save everything ----------------- %%

    % create some stuff to save for evaluation script and plotting later on
    YPred_piecewise = YPred_new;


    if saveResults
        fprintf('\nSaving evaluation results to "%s"...\n', evaluationOutputFile);
        save(evaluationOutputFile, ...
             'mseVal','maeVal','mseValNm','maeValNm', ...
             'predInNm','trueInNm','errorsNm','errorsNorm', ...
             'E_hertz_actual','E_hertz_pred', ...
             'E_500nm_actual','E_500nm_pred', ...
             'badHertzCount','bad500Count', ...
             'hertzMSE','hertzMAE','hertzMAPE', ...
             'mod500nmMSE','mod500nmMAE','mod500nmMAPE', ...
             'Y_true', "YPred_piecewise")
        fprintf('Evaluation results saved.\n');
    end

    %% ----------------- Generate (Optional) Scatter Plots ----------------- %%
    figure('Name','Iterative CP - Pred vs Actual (Fraction)','NumberTitle','off');
    scatter(Y_true, YPred_new, 36,'b','filled'); hold on; grid on;
    lineMin = min([Y_true; YPred_new]);
    lineMax = max([Y_true; YPred_new]);
    plot([lineMin, lineMax], [lineMin, lineMax], 'k--','LineWidth',1.5);
    xlabel('Actual CP (fraction)'); ylabel('Predicted CP (fraction)');
    title(sprintf('Iterative CP: MSE=%.4f, MAE=%.4f', mseVal, maeVal));

    figure('Name','Iterative CP - Pred vs Actual (nm)','NumberTitle','off');
    scatter(trueInNm, predInNm, 36,'b','filled'); hold on; grid on;
    lineMin = min([trueInNm; predInNm]);
    lineMax = max([trueInNm; predInNm]);
    plot([lineMin, lineMax], [lineMin, lineMax], 'k--','LineWidth',1.5);
    xlabel('Actual CP (nm)'); ylabel('Predicted CP (nm)');
    title(sprintf('Iterative CP: MSE=%.2f nm^2, MAE=%.2f nm', mseValNm, maeValNm));

    % Hertz Modulus scatter
    validH = ~isnan(E_hertz_actual) & ~isnan(E_hertz_pred);
    if any(validH)
        figure('Name','Hertz Modulus - Actual vs Predicted','NumberTitle','off');
        scatter(E_hertz_actual(validH), E_hertz_pred(validH),36,'r','filled');
        hold on; grid on;
        minVal = min([E_hertz_actual(validH); E_hertz_pred(validH)]);
        maxVal = max([E_hertz_actual(validH); E_hertz_pred(validH)]);
        plot([minVal, maxVal],[minVal, maxVal],'k--','LineWidth',1.5);
        xlabel('Hertz Modulus (Actual) [kPa]');
        ylabel('Hertz Modulus (Pred) [kPa]');
        title(sprintf('MSE=%.3f, MAE=%.3f, MAPE=%.1f%%', ...
            hertzMSE, hertzMAE, hertzMAPE));
    end

    % 500 nm Modulus scatter
    valid500 = ~isnan(E_500nm_actual) & ~isnan(E_500nm_pred);
    if any(valid500)
        figure('Name','Modulus at 500 nm - Actual vs Pred','NumberTitle','off');
        scatter(E_500nm_actual(valid500), E_500nm_pred(valid500),36,'g','filled');
        hold on; grid on;
        minVal2 = min([E_500nm_actual(valid500); E_500nm_pred(valid500)]);
        maxVal2 = max([E_500nm_actual(valid500); E_500nm_pred(valid500)]);
        plot([minVal2, maxVal2],[minVal2, maxVal2],'k--','LineWidth',1.5);
        xlabel('Modulus at 500 nm (Actual) [kPa]');
        ylabel('Modulus at 500 nm (Pred) [kPa]');
        title(sprintf('MSE=%.3f, MAE=%.3f, MAPE=%.1f%%', ...
            mod500nmMSE, mod500nmMAE, mod500nmMAPE));
    end

    fprintf('\nEvaluation complete.\n');
end
