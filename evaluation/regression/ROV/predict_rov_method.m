function predict_rov_method()
% predict_rov_method
% Estimates contact points from AFM data using the Ratio of Variance (RoV) method.
% The RoV at a given point is calculated as the variance of the following n points divided
% by the variance of the preceding n points, for the deflection values.
% Allows for a range of n values to be set.
% Results are stored in a matrix of size [numCurves x num_n_intervals].
    clc;
    close all;

    %% ----------------- Configuration ----------------- %%
    % File containing processed AFM data (extension & deflection)
    newDataFile = "C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v6_sandbox\training\regression_processed_files\processed_features_for_regression_All.mat";

    % Output file for saving the estimated contact points
    predictionOutputFile = 'predicted_contact_points_rov.mat';

    % Range of n for interval sizes
    nRange = 2:35;

    % Fraction of the curve to use (based on deflection values)
    dataFraction = 0.1;

    % Smoothing parameter (moving average window size)
    smoothingWindow = 4;

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
    % Matrix to store predicted contact points for each n in nRange
    YPredMatrix = nan(numCurves, length(nRange));

    %% ----------------- Process Each Curve ----------------- %%
    for i = 1:numCurves

        % 1) Extract curve data
        deflRaw = processedDefl(:, i);

        % Guard against degenerate (constant) data
        if range(deflRaw) < eps
            fprintf('Curve %d is near-constant. Skipping...\n', i);
            continue;
        end

        % 2) Apply smoothing to deflection curve
        if smoothingWindow > 0
            deflSmooth = movmean(deflRaw, smoothingWindow);
        else
            deflSmooth = deflRaw;
        end

        % 3) Use only the fraction of the curve specified by dataFraction (based on deflection)
        maxDefl = max(deflSmooth);
        threshold = dataFraction * maxDefl;
        validIdx = find(deflSmooth <= threshold);
        if isempty(validIdx)
            fprintf('Curve %d has no valid points below threshold. Skipping...\n', i);
            continue;
        end

        deflSmooth = deflSmooth(validIdx);
        curveLength = length(deflSmooth);

        parfor nIdx = 1:length(nRange)
            n = nRange(nIdx);

            % Ensure there are enough points for the calculation
            if curveLength < 2 * n + 1
                fprintf('Curve %d has insufficient points for n = %d. Skipping...\n', i, n);
                continue;
            end

            % 4) Calculate RoV for all valid points
            RoV = nan(curveLength, 1);
            for j = n + 1:curveLength - n
                varBefore = var(deflSmooth(j - n:j - 1));
                varAfter = var(deflSmooth(j + 1:j + n));

                if varBefore > 0 % Avoid division by zero
                    RoV(j) = varAfter / varBefore;
                end
            end

            % 5) Find the contact point based on minimum RoV
            [~, contactIdx] = min(RoV);

            % Store the normalized contact point
            YPredMatrix(i, nIdx) = (contactIdx - 1) / (numPoints - 1);
        end
    end

    %% ----------------- Save Results ----------------- %%
    fprintf('Saving predictions to "%s"...\n', predictionOutputFile);
    save(predictionOutputFile, 'YPredMatrix', 'nRange');
    fprintf('Predictions saved successfully to "%s".\n', predictionOutputFile);

    %% ----------------- Summary ----------------- %%
    validPreds = ~all(isnan(YPredMatrix), 2);
    fprintf('\nPredicted contact points for %d out of %d curves.\n', sum(validPreds), numCurves);

    %% ----------------- Plot Example ----------------- %%
    fprintf('Plotting example RoV curve...\n');
    exampleCurveIdx = find(validPreds, 1);
    if ~isempty(exampleCurveIdx)
        exampleDefl = processedDefl(:, exampleCurveIdx);
        exampleSmooth = movmean(exampleDefl, smoothingWindow);
        validIdx = find(exampleSmooth <= threshold);
        exampleSmooth = exampleSmooth(validIdx);
        exampleRoV = nan(length(exampleSmooth), 1);

        n = nRange(1); % Example interval size
        for j = n + 1:length(exampleSmooth) - n
            varBefore = var(exampleSmooth(j - n:j - 1));
            varAfter = var(exampleSmooth(j + 1:j + n));
            if varBefore > 0
                exampleRoV(j) = varAfter / varBefore;
            end
        end

        figure;
        plot(exampleSmooth, 'b', 'DisplayName', 'Smoothed Deflection'); hold on;
        plot(exampleRoV, 'r', 'DisplayName', 'RoV');
        legend;
        xlabel('Index');
        ylabel('Value');
        title(sprintf('RoV and Smoothed Deflection for Curve %d (n = %d)', exampleCurveIdx, n));
        grid on;
    end

    fprintf('RoV prediction process completed.\n');
end
