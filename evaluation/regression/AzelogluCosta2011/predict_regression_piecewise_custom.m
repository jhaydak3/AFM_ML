function predict_regression_piecewise_custom()
% predict_regression_piecewise_custom_deflectionFraction
% Estimates contact points from AFM data using a piecewise fit,
% but only including the portion of the curve where normalized deflection <= dataFraction.
%
% Usage:
%   1. Update 'newDataFile' path to point to the .mat file containing
%      'processedExt' and 'processedDefl'.
%   2. Run this function.
%
% The script:
%   - Loads processed AFM data from .mat file (extension & deflection).
%   - Normalizes each curve to [0,1].
%   - Truncates the data so that only points with normalized DEFLECTION <= dataFraction are included.
%   - Fits a piecewise model to estimate the contact point.
%   - Saves and plots the estimated contact points distribution.
%

% -------------------------------------------------------------------------

    close all;

    %% ----------------- Configuration ----------------- %%
    % File containing processed AFM data (extension & deflection)
    newDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\training\regression_processed_files\processed_features_for_regression_All.mat";

    % Output file for saving the estimated contact points
    predictionOutputFile = 'predicted_contact_points_piecewise_fit_deflectionFraction.mat';

    % Fraction of normalized DEFLECTION to include in the fit
    % e.g. 0.1 means only points where normalizedDeflection <= 0.1
    dataFraction = 0.3;   
    
    % Fitting parameter bounds
    % [ c,         linSlope,  polyExp,    polyAmp,   polyAmp2, linConstant, polyExp2 ]
    lb = [ 0,      -0.5,     2,          0,        0,          0,        1];   
    ub = [ 1,       0.5,     2,          500,       500          0,         1];
    
    % Options for lsqcurvefit
    fitOptions = optimoptions('lsqcurvefit',...
        'Display','off',...
        'MaxFunctionEvaluations',1e6,...
        'MaxIterations',1e6,'FunctionTolerance',1e-10, 'UseParallel',false);
    
    %% ----------------- Load New Data ----------------- %%
    fprintf('Loading new data from "%s"...\n', newDataFile);
    if ~isfile(newDataFile)
        error('Data file "%s" not found.', newDataFile);
    end
    newData = load(newDataFile);

    % We expect 'processedExt' and 'processedDefl'
    if ~isfield(newData, 'processedExt') || ~isfield(newData, 'processedDefl')
        error('Missing "processedExt" or "processedDefl" in "%s".', newDataFile);
    end
    
    processedExt  = newData.processedExt;    % [N x M] extension
    processedDefl = newData.processedDefl;   % [N x M] deflection
    [numPoints, numCurves] = size(processedExt);

    fprintf('Data loaded. Found %d curves, each with %d points.\n', numCurves, numPoints);

    %% ----------------- Allocate Arrays ----------------- %%
    YPred_new = nan(numCurves, 1);  % store contact points (predictions)

    %% ----------------- Fit Each Curve ----------------- %%
    parfor i = 1:numCurves
        i
        
        normDefl = processedDefl(:,i);
        normExt = processedExt(:,i);

        
        % 3) Truncate data where normalized DEFLECTION <= dataFraction
        validIdx = (normDefl <= dataFraction);
        xData = normExt(validIdx);
        yData = normDefl(validIdx);
        
        % If there's insufficient data in that range, skip
        if length(xData) < 5
            fprintf('Curve %d has insufficient points with deflection <= %.3f. Skipping...\n', i, dataFraction);
            continue;
        end

        % SET THE NEW UB
        thisUb = ub;
        thisUb(1) = xData(end);
        
        % 4) Set up initial guess (midpoints of bounds)
        p0 = 0.5*(lb + thisUb);

        % 5) Perform the fit using lsqcurvefit
        [pBest, ~, residual, exitflag] = lsqcurvefit( ...
            @piecewiseFun, ...  % piecewise model
            p0,                 ...  % initial guess
            xData,              ...  % xdata
            yData,              ...  % ydata
            lb, thisUb, fitOptions);
        
        if exitflag <= 0
            fprintf('Fit did not converge for curve %d.\n', i);
            continue;
        end
        
        % 6) Contact point is pBest(1)
        YPred_new(i) = pBest(1);

        % (Optional) Debug print:
        % fprintf('Curve %d: c=%.3f slope=%.3f exp=%.3f amp=%.3f amp2=%.3f (res=%.3g)\n', ...
        %     i, pBest(1), pBest(2), pBest(3), pBest(4), pBest(5), sum(residual.^2));
    end

    %% ----------------- Post-Processing ----------------- %%
    % Constrain predictions to [0,1] just in case
    YPred_new = max(min(YPred_new, 1), 0);

    % Display summary
    validPreds = ~isnan(YPred_new);
    fprintf('\nPredicted contact points for %d out of %d curves.\n', sum(validPreds), numCurves);

    %% ----------------- Save Results ----------------- %%
    fprintf('Saving predictions to "%s"...\n', predictionOutputFile);
    save(predictionOutputFile, 'YPred_new');
    fprintf('Predictions saved successfully to "%s".\n', predictionOutputFile);

    %% ----------------- Plot Distribution ----------------- %%
    fprintf('Plotting predicted contact points distribution...\n');
    figure('Name','Piecewise Fit Contact Points','NumberTitle','off');
    histogram(YPred_new(validPreds), 'Normalization','pdf',...
        'BinWidth',0.02, 'FaceColor','b', 'EdgeColor','k');
    xlabel('Predicted Contact Point (Normalized)');
    ylabel('Probability Density');
    title('Predicted Contact Points - Piecewise Fit (Deflection Fraction)');
    grid on;
    saveas(gcf, 'Predicted_Contact_Points_Piecewise_Fit_DeflectionFraction.png');
    fprintf('Plot saved as "Predicted_Contact_Points_Piecewise_Fit_DeflectionFraction.png".\n');

    fprintf('Piecewise fitting process completed.\n');
end


%% ----------------- Local Function ----------------- %%
function fvals = piecewiseFun(params, x)
% piecewiseFun:
%   params(1) = c (contact point)
%   params(2) = linSlope
%   params(3) = polyExp
%   params(4) = polyAmp
%   params(5) = polyAmp2
%   params(6) = linConstant
%
% For x <= c:
%   f(x) = linSlope * x + linConstant
%
% For x > c:
%   f(x) = linConstant + linSlope*c + polyAmp*(x-c)^polyExp +
%   polyAmp2*(x-c)^polyExp2

    cVal        = params(1);
    linSlope    = params(2);
    polyExp     = params(3);
    polyAmp     = params(4);
    polyAmp2    = params(5);
    linConstant = params(6);
    polyExp2    = params(7);

    fvals = zeros(size(x));

    idxLin  = (x <= cVal);
    idxPoly = (x >  cVal);

    % Linear portion
    fvals(idxLin) = linSlope .* x(idxLin) + linConstant;

    % Polynomial portion
    if any(idxPoly)
        xPoly  = x(idxPoly);
        fAtC   = linSlope*cVal + linConstant;  % continuity at x = c
        fvals(idxPoly) = fAtC + polyAmp.*(xPoly - cVal).^polyExp + ...
                                 polyAmp2.*(xPoly - cVal).^polyExp2;
    end
end
